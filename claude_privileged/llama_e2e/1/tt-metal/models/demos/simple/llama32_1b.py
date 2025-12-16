#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Simple Llama 3.2 1B implementation in ttnn.

This is a straightforward implementation showing how to bring up a model in ttnn.
It loads meta-llama/Llama-3.2-1B from HuggingFace, converts weights to ttnn,
and runs both prefill and decode passes using HuggingFace's generate function.

Usage:
    python llama32_1b.py --prompt "1 2 3 4 5 6 7 8 9 10 11 12"
"""

import argparse
import math
import torch
import torch.nn as nn
import ttnn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Cache, GenerationMixin, PreTrainedModel
from typing import Optional, Tuple


class TtnnLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Llama 3.2 1B model with HuggingFace compatibility, using ttnn for computation.

    This class wraps the HuggingFace Llama model and replaces forward passes
    with ttnn operations while keeping the same interface for generation.
    """

    def __init__(self, hf_model, ttnn_device):
        # Initialize PreTrainedModel with config
        super().__init__(hf_model.config)

        self.hf_model = hf_model
        self.config = hf_model.config
        self.ttnn_device = ttnn_device
        self.generation_config = hf_model.generation_config

        # Store HuggingFace model components for hybrid approach
        self.model = hf_model.model

        # Convert weights to ttnn lazily (on first use)
        self._ttnn_weights_loaded = False
        self._ttnn_embed_tokens = None
        self._ttnn_lm_head = None
        self._ttnn_layer_weights = []

    def _load_ttnn_weights(self):
        """Load and convert HuggingFace weights to ttnn format."""
        if self._ttnn_weights_loaded:
            return

        print("Converting weights to ttnn...")

        # Load embedding weights
        embed_weight = self.hf_model.model.embed_tokens.weight
        self._ttnn_embed_tokens = ttnn.from_torch(
            embed_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Load LM head weights (transpose for ttnn.linear)
        lm_head_weight = self.hf_model.lm_head.weight.T
        self._ttnn_lm_head = ttnn.from_torch(
            lm_head_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Load layer weights
        TILE_SIZE = 32
        for layer_idx, hf_layer in enumerate(self.hf_model.model.layers):
            print(f"Loading layer {layer_idx + 1}/{len(self.hf_model.model.layers)}...")

            layer_weights = {}

            # Attention weights (transpose for ttnn.linear)
            layer_weights['q_proj'] = ttnn.from_torch(
                hf_layer.self_attn.q_proj.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            layer_weights['k_proj'] = ttnn.from_torch(
                hf_layer.self_attn.k_proj.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            layer_weights['v_proj'] = ttnn.from_torch(
                hf_layer.self_attn.v_proj.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            layer_weights['o_proj'] = ttnn.from_torch(
                hf_layer.self_attn.o_proj.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # MLP weights (transpose for ttnn.linear)
            layer_weights['gate_proj'] = ttnn.from_torch(
                hf_layer.mlp.gate_proj.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            layer_weights['up_proj'] = ttnn.from_torch(
                hf_layer.mlp.up_proj.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            layer_weights['down_proj'] = ttnn.from_torch(
                hf_layer.mlp.down_proj.weight.T,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # RMSNorm weights (reshape to [1, 1, dim // TILE_SIZE, TILE_SIZE])
            layer_weights['input_layernorm'] = ttnn.from_torch(
                hf_layer.input_layernorm.weight.reshape(1, 1, self.config.hidden_size // TILE_SIZE, TILE_SIZE),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            layer_weights['post_attention_layernorm'] = ttnn.from_torch(
                hf_layer.post_attention_layernorm.weight.reshape(1, 1, self.config.hidden_size // TILE_SIZE, TILE_SIZE),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            self._ttnn_layer_weights.append(layer_weights)

        # Final norm
        self._ttnn_final_norm = ttnn.from_torch(
            self.hf_model.model.norm.weight.reshape(1, 1, self.config.hidden_size // TILE_SIZE, TILE_SIZE),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._ttnn_weights_loaded = True
        print("Weights loaded successfully!")

    def _apply_rotary_pos_emb_torch(self, q, k, cos, sin):
        """Apply rotary embeddings using torch (before converting to ttnn)."""
        # q, k: [batch, num_heads, seq_len, head_dim]
        # cos, sin: [seq_len, head_dim]

        # Reshape q and k to apply rotation
        q_embed = (q * cos.unsqueeze(0).unsqueeze(0)) + (self._rotate_half(q) * sin.unsqueeze(0).unsqueeze(0))
        k_embed = (k * cos.unsqueeze(0).unsqueeze(0)) + (self._rotate_half(k) * sin.unsqueeze(0).unsqueeze(0))

        return q_embed, k_embed

    def _rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _ttnn_layer_forward(self, hidden_states_ttnn, layer_idx, position_ids, past_key_value=None):
        """Forward pass for a single decoder layer using ttnn."""
        layer_weights = self._ttnn_layer_weights[layer_idx]
        hf_layer = self.hf_model.model.layers[layer_idx]

        # Convert to torch for operations not yet in ttnn
        hidden_states = ttnn.to_torch(hidden_states_ttnn)
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Self-attention
        residual = hidden_states

        # Input layernorm - use ttnn
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states_ttnn = ttnn.rms_norm(
            hidden_states_ttnn,
            epsilon=self.config.rms_norm_eps,
            weight=layer_weights['input_layernorm'],
        )
        hidden_states = ttnn.to_torch(hidden_states_ttnn)

        # Q, K, V projections using ttnn
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        query_states_ttnn = ttnn.linear(hidden_states_ttnn, layer_weights['q_proj'], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        key_states_ttnn = ttnn.linear(hidden_states_ttnn, layer_weights['k_proj'], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        value_states_ttnn = ttnn.linear(hidden_states_ttnn, layer_weights['v_proj'], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Convert to torch for reshape and RoPE
        query_states = ttnn.to_torch(query_states_ttnn)
        key_states = ttnn.to_torch(key_states_ttnn)
        value_states = ttnn.to_torch(value_states_ttnn)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.config.num_attention_heads, self.config.hidden_size // self.config.num_attention_heads)
        key_states = key_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.config.hidden_size // self.config.num_attention_heads)
        value_states = value_states.view(batch_size, seq_len, self.config.num_key_value_heads, self.config.hidden_size // self.config.num_attention_heads)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply RoPE using HuggingFace's implementation
        # rotary_emb is stored in the model, not in the layer
        cos, sin = self.hf_model.model.rotary_emb(value_states, position_ids)
        query_states, key_states = self._apply_rotary_pos_emb_torch(query_states, key_states, cos, sin)

        # Repeat K/V for GQA
        if self.config.num_key_value_heads != self.config.num_attention_heads:
            key_states = torch.repeat_interleave(key_states, self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)
            value_states = torch.repeat_interleave(value_states, self.config.num_attention_heads // self.config.num_key_value_heads, dim=1)

        # Attention computation using ttnn
        # Use PyTorch's scaled_dot_product_attention for now as it's simpler
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True if seq_len > 1 else False,
        )

        # Transpose back and reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)

        # Output projection using ttnn
        attn_output_ttnn = ttnn.from_torch(
            attn_output,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output_ttnn = ttnn.linear(attn_output_ttnn, layer_weights['o_proj'], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_output = ttnn.to_torch(attn_output_ttnn)

        # Residual connection
        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states

        # Post-attention layernorm using ttnn
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden_states_ttnn = ttnn.rms_norm(
            hidden_states_ttnn,
            epsilon=self.config.rms_norm_eps,
            weight=layer_weights['post_attention_layernorm'],
        )
        hidden_states = ttnn.to_torch(hidden_states_ttnn)

        # MLP forward using ttnn
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Gate and up projections
        gate_proj_ttnn = ttnn.linear(hidden_states_ttnn, layer_weights['gate_proj'], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up_proj_ttnn = ttnn.linear(hidden_states_ttnn, layer_weights['up_proj'], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # SiLU activation on gate
        gate_proj_ttnn = ttnn.silu(gate_proj_ttnn)

        # Element-wise multiply
        hidden_ttnn = ttnn.mul(gate_proj_ttnn, up_proj_ttnn)

        # Down projection
        hidden_ttnn = ttnn.linear(hidden_ttnn, layer_weights['down_proj'], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden_states = ttnn.to_torch(hidden_ttnn)

        # Residual connection
        hidden_states = residual + hidden_states

        # Convert back to ttnn for next layer
        hidden_states_ttnn = ttnn.from_torch(
            hidden_states,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return hidden_states_ttnn

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass compatible with HuggingFace generation.
        """
        # Load ttnn weights if not already loaded
        self._load_ttnn_weights()

        batch_size, seq_len = input_ids.shape

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embedding lookup using ttnn
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        hidden_states_ttnn = ttnn.embedding(input_ids_ttnn, self._ttnn_embed_tokens, layout=ttnn.TILE_LAYOUT)

        # Pass through decoder layers
        for layer_idx in range(len(self.hf_model.model.layers)):
            hidden_states_ttnn = self._ttnn_layer_forward(
                hidden_states_ttnn,
                layer_idx,
                position_ids,
                past_key_value=past_key_values[layer_idx] if past_key_values else None,
            )

        # Final norm using ttnn
        hidden_states_ttnn = ttnn.rms_norm(
            hidden_states_ttnn,
            epsilon=self.config.rms_norm_eps,
            weight=self._ttnn_final_norm,
        )

        # LM head using ttnn
        logits_ttnn = ttnn.linear(hidden_states_ttnn, self._ttnn_lm_head, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Convert to torch for output
        logits = ttnn.to_torch(logits_ttnn)

        # Return in HuggingFace format
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,  # Not implementing KV cache for simplicity
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation (required by HuggingFace generate)."""
        return {"input_ids": input_ids}

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
        """Update model kwargs for generation (required by HuggingFace generate)."""
        return model_kwargs

    def can_generate(self):
        """Check if model can generate (required by HuggingFace)."""
        return True


def main():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 1B on ttnn")
    parser.add_argument(
        "--prompt",
        type=str,
        default="1 2 3 4 5 6 7 8 9 10 11 12",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=15,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID to use",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Llama 3.2 1B on ttnn")
    print("=" * 80)

    # Initialize device
    print("\nInitializing ttnn device...")
    device = ttnn.open_device(device_id=args.device_id)

    try:
        # Load HuggingFace model and tokenizer
        model_name = "meta-llama/Llama-3.2-1B"
        print(f"\nLoading HuggingFace model: {model_name}")

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create ttnn model
        print("\nCreating ttnn model...")
        ttnn_model = TtnnLlamaForCausalLM(hf_model, device)

        # Prepare input
        print(f"\nPrompt: {args.prompt}")
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Generate with ttnn model using HuggingFace's generate
        print("\nGenerating with ttnn model...")
        with torch.no_grad():
            output_ids = ttnn_model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\nGenerated text:\n{generated_text}")

        # For comparison, generate with HuggingFace model
        print("\n" + "=" * 80)
        print("Comparison with HuggingFace model:")
        print("=" * 80)

        with torch.no_grad():
            hf_outputs = hf_model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hf_generated_text = tokenizer.decode(hf_outputs[0], skip_special_tokens=True)
        print(f"\nHuggingFace generated text:\n{hf_generated_text}")

        print("\n" + "=" * 80)
        print("Done!")
        print("=" * 80)

    finally:
        # Clean up
        print("\nClosing device...")
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
