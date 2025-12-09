#!/usr/bin/env python3
"""
Simple Llama 3.2 1B implementation in ttnn.

This is a clean, educational implementation showing how to port a Hugging Face model to ttnn.
The model is compatible with Hugging Face's generate() function for elegant inference.

Usage:
    python llama32_1b.py
"""

import torch
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Optional, Tuple
import argparse


def apply_rotary_emb_huggingface_format(x, freqs_cos, freqs_sin):
    """
    Apply rotary embeddings in HuggingFace format (not Meta format).

    HuggingFace format keeps cos/sin coefficients separate and doesn't require
    weight swizzling like Meta format does.

    Args:
        x: Input tensor of shape [..., seq_len, n_heads, head_dim]
        freqs_cos: Cosine frequencies
        freqs_sin: Sine frequencies

    Returns:
        Tensor with rotary embeddings applied
    """
    # Split the last dimension in half for rotation
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = ttnn.subtract(
        ttnn.multiply(x1, freqs_cos),
        ttnn.multiply(x2, freqs_sin)
    )
    out2 = ttnn.add(
        ttnn.multiply(x1, freqs_sin),
        ttnn.multiply(x2, freqs_cos)
    )

    return ttnn.concat([out1, out2], dim=-1)


class LlamaRMSNorm:
    """RMSNorm normalization layer."""

    def __init__(self, weight, eps=1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, hidden_states):
        """
        Apply RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        """
        # Calculate variance (mean of squares)
        variance = ttnn.mean(ttnn.pow(hidden_states, 2.0), dim=-1, keepdim=True)

        # Add epsilon and take reciprocal square root
        hidden_states = ttnn.multiply(
            hidden_states,
            ttnn.rsqrt(ttnn.add(variance, self.eps))
        )

        # Apply learned weight
        return ttnn.multiply(hidden_states, self.weight)


class LlamaAttention:
    """Multi-head attention with rotary embeddings."""

    def __init__(self, config, layer_idx, q_proj, k_proj, v_proj, o_proj, device):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.device = device

        # Projection weights (note: ttnn.linear expects transposed weights)
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj

        # Pre-compute rotary embeddings
        self._init_rope()

    def _init_rope(self):
        """Initialize rotary position embeddings."""
        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )

        # Create position indices
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)

        # Compute frequencies: outer product of positions and inv_freq
        freqs = torch.outer(t, inv_freq)

        # Duplicate for both halves of head_dim (HuggingFace format)
        emb = torch.cat([freqs, freqs], dim=-1)

        # Convert to ttnn and store cos/sin
        self.cos_cached = ttnn.from_torch(
            emb.cos()[None, None, :, :],
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.sin_cached = ttnn.from_torch(
            emb.sin()[None, None, :, :],
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass of attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE

        Returns:
            Attention output: [batch_size, seq_len, hidden_size]
        """
        bsz, seq_len, _ = hidden_states.shape

        # QKV projections
        # Note: ttnn.linear expects (A,B) @ (B,C) -> (A,C)
        query_states = ttnn.linear(hidden_states, self.q_proj)
        key_states = ttnn.linear(hidden_states, self.k_proj)
        value_states = ttnn.linear(hidden_states, self.v_proj)

        # Reshape to separate heads: [bsz, seq_len, n_heads, head_dim]
        query_states = ttnn.reshape(query_states, (bsz, seq_len, self.num_heads, self.head_dim))
        key_states = ttnn.reshape(key_states, (bsz, seq_len, self.num_key_value_heads, self.head_dim))
        value_states = ttnn.reshape(value_states, (bsz, seq_len, self.num_key_value_heads, self.head_dim))

        # Apply rotary embeddings
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        query_states = apply_rotary_emb_huggingface_format(query_states, cos, sin)
        key_states = apply_rotary_emb_huggingface_format(key_states, cos, sin)

        # Grouped-query attention: repeat k/v heads if needed
        if self.num_key_value_groups > 1:
            key_states = ttnn.repeat_interleave(key_states, self.num_key_value_groups, dim=2)
            value_states = ttnn.repeat_interleave(value_states, self.num_key_value_groups, dim=2)

        # Transpose for attention: [bsz, n_heads, seq_len, head_dim]
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states = ttnn.permute(key_states, (0, 2, 1, 3))
        value_states = ttnn.permute(value_states, (0, 2, 1, 3))

        # Scaled dot-product attention using ttnn
        # This is more efficient than manual implementation
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            attention_mask=attention_mask
        )

        # Reshape back: [bsz, seq_len, n_heads * head_dim]
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (bsz, seq_len, self.hidden_size))

        # Output projection
        attn_output = ttnn.linear(attn_output, self.o_proj)

        return attn_output


class LlamaMLP:
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, gate_proj, up_proj, down_proj):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def __call__(self, x):
        """
        SwiGLU activation: gate_proj(x) * silu(up_proj(x))
        Then project down with down_proj
        """
        gate = ttnn.linear(x, self.gate_proj)
        up = ttnn.linear(x, self.up_proj)

        # SwiGLU: silu(gate) * up
        gate = ttnn.silu(gate)
        hidden = ttnn.multiply(gate, up)

        # Down projection
        output = ttnn.linear(hidden, self.down_proj)

        return output


class LlamaDecoderLayer:
    """Single transformer decoder layer."""

    def __init__(self, config, layer_idx, hf_layer, device):
        self.layer_idx = layer_idx

        # Convert and transpose weights for ttnn.linear
        # ttnn.linear expects (B,C) not (C,B) like torch
        q_weight = hf_layer.self_attn.q_proj.weight.T
        k_weight = hf_layer.self_attn.k_proj.weight.T
        v_weight = hf_layer.self_attn.v_proj.weight.T
        o_weight = hf_layer.self_attn.o_proj.weight.T
        gate_weight = hf_layer.mlp.gate_proj.weight.T
        up_weight = hf_layer.mlp.up_proj.weight.T
        down_weight = hf_layer.mlp.down_proj.weight.T

        # Convert to ttnn
        q_proj = ttnn.from_torch(q_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_proj = ttnn.from_torch(k_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_proj = ttnn.from_torch(v_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o_proj = ttnn.from_torch(o_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_proj = ttnn.from_torch(gate_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up_proj = ttnn.from_torch(up_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down_proj = ttnn.from_torch(down_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Initialize sub-modules
        self.self_attn = LlamaAttention(config, layer_idx, q_proj, k_proj, v_proj, o_proj, device)
        self.mlp = LlamaMLP(gate_proj, up_proj, down_proj)

        # Layer norms
        input_norm_weight = ttnn.from_torch(
            hf_layer.input_layernorm.weight,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        post_attn_norm_weight = ttnn.from_torch(
            hf_layer.post_attention_layernorm.weight,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        self.input_layernorm = LlamaRMSNorm(input_norm_weight, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(post_attn_norm_weight, eps=config.rms_norm_eps)

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        """
        Forward pass with pre-norm residual connections.
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = ttnn.add(residual, hidden_states)

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states


class TtnnLlamaForCausalLM:
    """
    Llama model for causal language modeling, compatible with HuggingFace generate().

    This class wraps the ttnn implementation to work seamlessly with HuggingFace's
    generation utilities.
    """

    def __init__(self, model_name="meta-llama/Llama-3.2-1B", device=None):
        # Load reference model and config
        print(f"Loading reference model: {model_name}")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.config = self.hf_model.config

        # Initialize ttnn device
        if device is None:
            device = ttnn.open_device(device_id=0)
        self.device = device

        print("Converting weights to ttnn...")
        self._convert_weights()

        # Store necessary attributes for HuggingFace compatibility
        self.generation_config = GenerationConfig.from_model_config(self.config)

    def _convert_weights(self):
        """Convert HuggingFace weights to ttnn format."""
        # Embedding layer
        embed_weight = self.hf_model.model.embed_tokens.weight
        self.embed_tokens = ttnn.from_torch(
            embed_weight,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Transformer layers
        self.layers = []
        for layer_idx, hf_layer in enumerate(self.hf_model.model.layers):
            print(f"  Converting layer {layer_idx + 1}/{len(self.hf_model.model.layers)}")
            layer = LlamaDecoderLayer(self.config, layer_idx, hf_layer, self.device)
            self.layers.append(layer)

        # Final norm
        norm_weight = ttnn.from_torch(
            self.hf_model.model.norm.weight,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.norm = LlamaRMSNorm(norm_weight, eps=self.config.rms_norm_eps)

        # LM head (transposed for ttnn.linear)
        lm_head_weight = self.hf_model.lm_head.weight.T
        self.lm_head = ttnn.from_torch(
            lm_head_weight,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        print("Weight conversion complete!")

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        """
        Forward pass compatible with HuggingFace.

        Returns:
            Object with 'logits' attribute containing output logits
        """
        # Convert input_ids to ttnn if needed
        if isinstance(input_ids, torch.Tensor):
            # Embedding lookup
            input_ids_ttnn = ttnn.from_torch(
                input_ids,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            hidden_states = ttnn.embedding(input_ids_ttnn, self.embed_tokens)
        else:
            hidden_states = ttnn.embedding(input_ids, self.embed_tokens)

        # Process through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head to get logits
        logits = ttnn.linear(hidden_states, self.lm_head)

        # Convert back to torch for HuggingFace compatibility
        logits_torch = ttnn.to_torch(logits)

        # Return object with logits attribute (HuggingFace expects this)
        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(logits_torch)

    def __call__(self, *args, **kwargs):
        """Make the model callable."""
        return self.forward(*args, **kwargs)

    def generate(self, input_ids, **kwargs):
        """
        Generate text using HuggingFace's generate function.

        This delegates to the HuggingFace generate() implementation which
        will call our forward() method internally.
        """
        from transformers.generation import GenerationMixin

        # Temporarily make this instance a GenerationMixin
        class GeneratableTtnnLlama(GenerationMixin):
            def __init__(self, ttnn_model):
                self.ttnn_model = ttnn_model
                self.config = ttnn_model.config
                self.generation_config = ttnn_model.generation_config

            def forward(self, *args, **kwargs):
                return self.ttnn_model.forward(*args, **kwargs)

            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                return {"input_ids": input_ids}

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

        gen_model = GeneratableTtnnLlama(self)
        return gen_model.generate(input_ids, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation."""
        return {"input_ids": input_ids}


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Llama 3.2 1B ttnn demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="1 2 3 4 5 6 7 8 9 10 11 12",
        help="Input prompt for generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Llama 3.2 1B in ttnn - Simple Demo")
    print("=" * 80)

    # Initialize model
    print("\n[1/4] Initializing ttnn device...")
    device = ttnn.open_device(device_id=0)

    print("\n[2/4] Loading model and converting to ttnn...")
    model = TtnnLlamaForCausalLM(model_name=args.model_name, device=device)

    print("\n[3/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare input
    print("\n[4/4] Running generation...")
    print(f"\nPrompt: {args.prompt}")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")

    # Generate using HuggingFace's generate function
    print("\nGenerating...")
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,  # Greedy decoding for deterministic output
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and print output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nOutput: {output_text}")

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)

    # Cleanup
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
