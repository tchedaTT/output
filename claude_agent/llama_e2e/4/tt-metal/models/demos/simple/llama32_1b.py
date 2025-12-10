#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Simple implementation of Llama 3.2 1B in ttnn.

This module loads meta-llama/Llama-3.2-1B from HuggingFace, converts the weights to ttnn,
and runs both prefill and decode passes to generate output for a given prompt.

Usage:
    python llama32_1b.py "Your prompt here"
"""

import sys
import math
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import ttnn


def apply_rotary_emb_torch(q, k, cos, sin):
    """
    Apply RoPE using torch (HuggingFace format).

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine values [1, 1, seq_len, head_dim]
        sin: Sine values [1, 1, seq_len, head_dim]
    """
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Expand cos/sin to match q/k batch dimension
    cos = cos.expand(q.shape[0], -1, -1, -1)
    sin = sin.expand(q.shape[0], -1, -1, -1)

    # Apply rotation: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Llama attention module."""

    def __init__(self, config, layer_idx, device):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Compute RoPE cos/sin matrices (HuggingFace format)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)

        # Weights (to be loaded)
        self.wqkv = None
        self.wo = None

    def load_weights(self, state_dict, layer_num):
        """Load weights from state dict."""
        prefix = f"model.layers.{layer_num}.self_attn"

        # Load Q, K, V weights
        wq = state_dict[f"{prefix}.q_proj.weight"]
        wk = state_dict[f"{prefix}.k_proj.weight"]
        wv = state_dict[f"{prefix}.v_proj.weight"]

        # Concatenate and transpose for ttnn.linear
        wqkv = torch.cat([wq, wk, wv], dim=0).T  # [hidden_size, total_qkv_size]

        self.wqkv = ttnn.from_torch(
            wqkv,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Load output projection
        wo = state_dict[f"{prefix}.o_proj.weight"].T
        self.wo = ttnn.from_torch(
            wo,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward(self, x_ttnn, position_ids, is_prefill=True):
        """Forward pass."""
        # QKV projection
        xqkv = ttnn.linear(x_ttnn, self.wqkv)

        # Convert to torch for splitting and reshaping
        # (This is a simplification - could be optimized with pure ttnn ops)
        xqkv_torch = ttnn.to_torch(xqkv)
        batch_size, seq_len = xqkv_torch.shape[0], xqkv_torch.shape[1]

        # Split into Q, K, V
        q_size = self.num_heads * self.head_dim
        k_size = self.num_key_value_heads * self.head_dim
        v_size = self.num_key_value_heads * self.head_dim

        q = xqkv_torch[..., :q_size]
        k = xqkv_torch[..., q_size:q_size + k_size]
        v = xqkv_torch[..., q_size + k_size:]

        # Reshape to separate heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q, k = apply_rotary_emb_torch(q, k, cos, sin)

        # Handle GQA: repeat K, V to match Q heads
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Convert back to ttnn for attention
        q_ttnn = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_ttnn = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_ttnn = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                  device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_ttnn, k_ttnn, v_ttnn,
            is_causal=True,
            scale=scale
        )

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_torch = ttnn.to_torch(attn_output)
        attn_torch = attn_torch.transpose(1, 2).contiguous()
        attn_torch = attn_torch.view(batch_size, seq_len, self.hidden_size)

        attn_ttnn = ttnn.from_torch(
            attn_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Output projection
        output = ttnn.linear(attn_ttnn, self.wo)

        return output


class LlamaMLP(nn.Module):
    """Llama MLP module."""

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Weights (to be loaded)
        self.w1 = None  # gate_proj
        self.w2 = None  # down_proj
        self.w3 = None  # up_proj

    def load_weights(self, state_dict, layer_num):
        """Load weights from state dict."""
        prefix = f"model.layers.{layer_num}.mlp"

        # Load and transpose for ttnn.linear
        w1 = state_dict[f"{prefix}.gate_proj.weight"].T
        w2 = state_dict[f"{prefix}.down_proj.weight"].T
        w3 = state_dict[f"{prefix}.up_proj.weight"].T

        self.w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.w3 = ttnn.from_torch(w3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, x):
        """
        Forward pass: down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        # gate_proj and silu activation
        gate = ttnn.linear(x, self.w1)
        gate = ttnn.silu(gate)

        # up_proj
        up = ttnn.linear(x, self.w3)

        # Multiply
        intermediate = ttnn.mul(gate, up)

        # down_proj
        output = ttnn.linear(intermediate, self.w2)

        return output


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer."""

    def __init__(self, config, layer_idx, device):
        super().__init__()
        self.layer_idx = layer_idx
        self.device = device

        self.self_attn = LlamaAttention(config, layer_idx, device)
        self.mlp = LlamaMLP(config, device)

        self.rms_norm_eps = config.rms_norm_eps

        # Norm weights (to be loaded)
        self.input_layernorm_weight = None
        self.post_attention_layernorm_weight = None

    def load_weights(self, state_dict, layer_num):
        """Load weights from state dict."""
        prefix = f"model.layers.{layer_num}"

        # Load attention and MLP weights
        self.self_attn.load_weights(state_dict, layer_num)
        self.mlp.load_weights(state_dict, layer_num)

        # Load norm weights
        input_norm = state_dict[f"{prefix}.input_layernorm.weight"]
        post_attn_norm = state_dict[f"{prefix}.post_attention_layernorm.weight"]

        self.input_layernorm_weight = ttnn.from_torch(
            input_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.post_attention_layernorm_weight = ttnn.from_torch(
            post_attn_norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward(self, hidden_states, position_ids, is_prefill=True):
        """Forward pass."""
        # Pre-attention norm
        residual = hidden_states
        hidden_states = ttnn.rms_norm(hidden_states, self.input_layernorm_weight, eps=self.rms_norm_eps)

        # Attention
        hidden_states = self.self_attn(hidden_states, position_ids, is_prefill)

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        # Pre-MLP norm
        residual = hidden_states
        hidden_states = ttnn.rms_norm(hidden_states, self.post_attention_layernorm_weight, eps=self.rms_norm_eps)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Residual connection
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states


class LlamaForCausalLM(nn.Module):
    """
    Llama model for causal language modeling.
    Compatible with HuggingFace's generate() function.
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.vocab_size = config.vocab_size

        # Embedding (to be loaded)
        self.embed_tokens_weight = None

        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, device)
            for i in range(config.num_hidden_layers)
        ])

        # Final norm and LM head (to be loaded)
        self.norm_weight = None
        self.lm_head_weight = None
        self.rms_norm_eps = config.rms_norm_eps

    def load_weights_from_hf(self, hf_model):
        """Load weights from HuggingFace model."""
        state_dict = hf_model.state_dict()

        # Load embedding
        embed = state_dict["model.embed_tokens.weight"]
        self.embed_tokens_weight = ttnn.from_torch(
            embed,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Load decoder layers
        for i, layer in enumerate(self.layers):
            print(f"Loading layer {i+1}/{len(self.layers)}...")
            layer.load_weights(state_dict, i)

        # Load final norm
        norm = state_dict["model.norm.weight"]
        self.norm_weight = ttnn.from_torch(
            norm,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Load LM head
        lm_head = state_dict["lm_head.weight"].T
        self.lm_head_weight = ttnn.from_torch(
            lm_head,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        """
        Forward pass - compatible with HuggingFace interface.
        """
        batch_size, seq_len = input_ids.shape

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        # Determine mode
        is_prefill = seq_len > 1

        # Embedding lookup
        input_ids_ttnn = ttnn.from_torch(
            input_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        hidden_states = ttnn.embedding(input_ids_ttnn, self.embed_tokens_weight, layout=ttnn.TILE_LAYOUT)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, is_prefill)

        # Final norm
        hidden_states = ttnn.rms_norm(hidden_states, self.norm_weight, eps=self.rms_norm_eps)

        # LM head
        logits = ttnn.linear(hidden_states, self.lm_head_weight)

        # Convert to torch
        logits_torch = ttnn.to_torch(logits)

        # Return in HuggingFace format
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            logits=logits_torch,
            past_key_values=past_key_values,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """Prepare inputs for generation."""
        if past_key_values is not None:
            # Only use last token in decode mode
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search."""
        return past_key_values


def main():
    """Main function."""
    # Get prompt
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "1 2 3 4 5 6 7 8 9 10 11 12"

    print(f"Prompt: {prompt}\n")

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Load HuggingFace model
        model_name = "meta-llama/Llama-3.2-1B"
        print(f"Loading {model_name} from HuggingFace...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

        print(f"\nModel config:")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  KV heads: {config.num_key_value_heads}")
        print(f"  Intermediate size: {config.intermediate_size}")
        print(f"  Vocab size: {config.vocab_size}\n")

        # Create ttnn model
        print("Converting model to ttnn...")
        model = LlamaForCausalLM(config, device)
        model.load_weights_from_hf(hf_model)
        print("Model loaded!\n")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        print(f"Input: {tokenizer.decode(input_ids[0])}")
        print(f"Input tokens: {input_ids.tolist()}\n")

        # Generate
        print("Generating...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nOutput: {output_text}")
        print(f"Output tokens: {outputs.tolist()}\n")

        print("Success!")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
