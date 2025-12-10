#!/usr/bin/env python3
"""
Simple Llama 3.2 1B implementation in ttnn.

This is a clean, educational implementation showing how to port a model to ttnn.
It loads meta-llama/Llama-3.2-1B from HuggingFace and runs inference on ttnn.

The implementation uses ttnn for the compute-intensive operations (linear layers, attention)
while keeping the control flow and less critical operations in PyTorch for simplicity.

Usage:
    python llama32_1b.py ["prompt text"]

Example:
    python llama32_1b.py "1 2 3 4 5 6 7 8 9 10 11 12"
"""

import sys
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import ttnn


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, rope_scaling: dict = None):
    """
    Precompute cos and sin frequencies for rotary embeddings.
    Uses the HuggingFace format.

    For Llama 3.2, rope_scaling includes:
        - factor: 32.0
        - low_freq_factor: 1.0
        - high_freq_factor: 4.0
        - original_max_position_embeddings: 8192
        - rope_type: "llama3"
    """
    # Compute base frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Apply Llama 3 scaling if provided
    if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
        factor = rope_scaling.get("factor", 1.0)
        low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
        freqs = torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    # Create position indices
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # Compute outer product
    freqs = torch.outer(t, freqs)

    # Compute cos and sin (HuggingFace repeats the frequencies)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.
    This uses the HuggingFace format (NOT Meta's interleaved format).

    Args:
        xq: Query tensor [batch, n_heads, seq_len, head_dim]
        xk: Key tensor [batch, n_kv_heads, seq_len, head_dim]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
    """
    # Split into first and second half for rotation
    xq_r, xq_i = xq[..., : xq.shape[-1] // 2], xq[..., xq.shape[-1] // 2 :]
    xk_r, xk_i = xk[..., : xk.shape[-1] // 2], xk[..., xk.shape[-1] // 2 :]

    # Reshape cos and sin to match (also split to match head_dim//2)
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(0).unsqueeze(0)

    # Apply rotation: (r, i) * (cos, sin) = (r*cos - i*sin, r*sin + i*cos)
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    xq_out = torch.cat([xq_out_r, xq_out_i], dim=-1)
    xk_out = torch.cat([xk_out_r, xk_out_i], dim=-1)

    return xq_out, xk_out


class LlamaAttention(nn.Module):
    """Llama attention module using ttnn for compute-intensive ops."""

    def __init__(self, config, layer_idx: int, device=None, state_dict=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.ttnn_device = device

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Load weights for ttnn
        if state_dict is not None and device is not None:
            prefix = f"model.layers.{layer_idx}.self_attn"

            # Note: ttnn.linear expects (A, B) @ (B, C) while torch expects (A, B) @ (C, B)
            # So we transpose the weights
            wq = state_dict[f"{prefix}.q_proj.weight"].T.unsqueeze(0).unsqueeze(0)
            wk = state_dict[f"{prefix}.k_proj.weight"].T.unsqueeze(0).unsqueeze(0)
            wv = state_dict[f"{prefix}.v_proj.weight"].T.unsqueeze(0).unsqueeze(0)
            wo = state_dict[f"{prefix}.o_proj.weight"].T.unsqueeze(0).unsqueeze(0)

            self.wq = ttnn.from_torch(wq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.wk = ttnn.from_torch(wk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.wv = ttnn.from_torch(wv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.wo = ttnn.from_torch(wo, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
    ):
        """
        Forward pass using ttnn for linear ops and torch for the rest.

        Args:
            x: Input tensor [batch, seqlen, hidden_size]
            cos, sin: Rotary embedding tensors
            start_pos: Starting position for RoPE
        """
        batch, seqlen, _ = x.shape

        # Convert to ttnn for linear projections
        x_tt = ttnn.from_torch(
            x.unsqueeze(0).unsqueeze(0),  # [1, 1, batch*seqlen, hidden_size]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # QKV projections using ttnn.linear
        xq_tt = ttnn.linear(x_tt, self.wq, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xk_tt = ttnn.linear(x_tt, self.wk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xv_tt = ttnn.linear(x_tt, self.wv, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Convert back to torch for reshape and RoPE
        xq = ttnn.to_torch(ttnn.from_device(xq_tt)).squeeze(0).squeeze(0)[:batch*seqlen]
        xk = ttnn.to_torch(ttnn.from_device(xk_tt)).squeeze(0).squeeze(0)[:batch*seqlen]
        xv = ttnn.to_torch(ttnn.from_device(xv_tt)).squeeze(0).squeeze(0)[:batch*seqlen]

        # Reshape for attention: [batch, seqlen, n_heads, head_dim] -> [batch, n_heads, seqlen, head_dim]
        xq = xq.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch, seqlen, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, cos[start_pos:start_pos+seqlen], sin[start_pos:start_pos+seqlen])

        # Grouped-query attention: repeat k/v heads if needed
        if self.num_key_value_groups > 1:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=1)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=1)

        # Convert to ttnn for scaled dot product attention
        xq_tt = ttnn.from_torch(
            xq,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xk_tt = ttnn.from_torch(
            xk,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xv_tt = ttnn.from_torch(
            xv,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Use ttnn's scaled_dot_product_attention
        scale = 1.0 / math.sqrt(self.head_dim)
        output_tt = ttnn.transformer.scaled_dot_product_attention(
            xq_tt, xk_tt, xv_tt,
            is_causal=True,
            scale=scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Convert back to torch and reshape
        output = ttnn.to_torch(ttnn.from_device(output_tt))
        output = output[:batch, :, :seqlen, :].transpose(1, 2).contiguous().view(batch, seqlen, -1)

        # Output projection using ttnn
        output_tt = ttnn.from_torch(
            output.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        output_tt = ttnn.linear(output_tt, self.wo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.to_torch(ttnn.from_device(output_tt)).squeeze(0).squeeze(0)[:batch*seqlen].view(batch, seqlen, -1)

        return output


class LlamaFeedForward(nn.Module):
    """Llama MLP/FeedForward module using ttnn."""

    def __init__(self, config, layer_idx: int, device=None, state_dict=None):
        super().__init__()
        self.config = config
        self.ttnn_device = device

        if state_dict is not None and device is not None:
            prefix = f"model.layers.{layer_idx}.mlp"

            # Transpose for ttnn.linear format
            w1 = state_dict[f"{prefix}.gate_proj.weight"].T.unsqueeze(0).unsqueeze(0)
            w2 = state_dict[f"{prefix}.down_proj.weight"].T.unsqueeze(0).unsqueeze(0)
            w3 = state_dict[f"{prefix}.up_proj.weight"].T.unsqueeze(0).unsqueeze(0)

            self.w1 = ttnn.from_torch(w1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.w2 = ttnn.from_torch(w2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.w3 = ttnn.from_torch(w3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, x: torch.Tensor):
        """Forward pass: SwiGLU activation."""
        batch, seqlen, hidden = x.shape

        # Convert to ttnn
        x_tt = ttnn.from_torch(
            x.view(batch * seqlen, hidden).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # SwiGLU: gate_proj(x).silu() * up_proj(x) @ down_proj
        gate_tt = ttnn.linear(x_tt, self.w1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_tt = ttnn.silu(gate_tt)

        up_tt = ttnn.linear(x_tt, self.w3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_tt = ttnn.mul(gate_tt, up_tt)
        output_tt = ttnn.linear(hidden_tt, self.w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Convert back to torch
        output = ttnn.to_torch(ttnn.from_device(output_tt)).squeeze(0).squeeze(0)[:batch*seqlen]
        output = output.view(batch, seqlen, hidden)

        return output


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder/transformer layer."""

    def __init__(self, config, layer_idx: int, device=None, state_dict=None):
        super().__init__()
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = LlamaAttention(config, layer_idx, device, state_dict)

        # MLP
        self.mlp = LlamaFeedForward(config, layer_idx, device, state_dict)

        # Layer norms (keep in PyTorch for simplicity)
        if state_dict is not None:
            prefix = f"model.layers.{layer_idx}"
            self.input_layernorm_weight = state_dict[f"{prefix}.input_layernorm.weight"]
            self.post_attention_layernorm_weight = state_dict[f"{prefix}.post_attention_layernorm.weight"]
        else:
            self.input_layernorm_weight = torch.ones(config.hidden_size)
            self.post_attention_layernorm_weight = torch.ones(config.hidden_size)

        self.norm_eps = config.rms_norm_eps

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor):
        """RMS normalization."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.norm_eps)
        return weight * x

    def forward(self, x, cos, sin, start_pos=0):
        # Pre-norm architecture with residual connections
        h = x + self.self_attn(
            self.rms_norm(x, self.input_layernorm_weight),
            cos, sin, start_pos
        )
        out = h + self.mlp(self.rms_norm(h, self.post_attention_layernorm_weight))
        return out


class LlamaModel(nn.Module):
    """Main Llama model."""

    def __init__(self, config, device=None, state_dict=None):
        super().__init__()
        self.config = config
        self.ttnn_device = device
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Embedding (keep in PyTorch)
        if state_dict is not None:
            self.embed_tokens = nn.Embedding.from_pretrained(state_dict["model.embed_tokens.weight"], freeze=True)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, device, state_dict)
            for i in range(config.num_hidden_layers)
        ])

        # Final norm (keep in PyTorch)
        if state_dict is not None:
            self.norm_weight = state_dict["model.norm.weight"]
        else:
            self.norm_weight = torch.ones(config.hidden_size)
        self.norm_eps = config.rms_norm_eps

        # Precompute rotary embeddings
        self.cos, self.sin = precompute_freqs_cis(
            config.head_dim if hasattr(config, 'head_dim') else config.hidden_size // config.num_attention_heads,
            config.max_position_embeddings * 2,
            config.rope_theta,
            config.rope_scaling if hasattr(config, 'rope_scaling') else None
        )

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor):
        """RMS normalization."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.norm_eps)
        return weight * x

    def forward(self, input_ids, start_pos=0):
        # Embedding
        h = self.embed_tokens(input_ids)

        # Pass through decoder layers
        for layer in self.layers:
            h = layer(h, self.cos, self.sin, start_pos)

        # Final norm
        h = self.rms_norm(h, self.norm_weight)

        return h


class LlamaForCausalLM(nn.Module):
    """Llama model for causal language modeling."""

    def __init__(self, config, device=None, state_dict=None):
        super().__init__()
        self.config = config
        self.ttnn_device = device
        self.model = LlamaModel(config, device, state_dict)

        # LM head (keep in PyTorch for simplicity at the boundary)
        if state_dict is not None:
            # Llama 3.2 1B uses tied weights
            if "lm_head.weight" in state_dict:
                lm_head_weight = state_dict["lm_head.weight"]
            else:
                lm_head_weight = state_dict["model.embed_tokens.weight"]
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight.data = lm_head_weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, start_pos=0):
        hidden_states = self.model(input_ids, start_pos)
        logits = self.lm_head(hidden_states)
        return type('Output', (), {'logits': logits})()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def can_generate(self):
        return True


def main():
    """Main function to run the demo."""
    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "1 2 3 4 5 6 7 8 9 10 11 12"

    print(f"Loading Llama 3.2 1B model...")
    print(f"Prompt: {prompt}")

    # Load model and tokenizer from HuggingFace
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Load reference model to get weights
    print("Loading reference model...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    state_dict = reference_model.state_dict()

    # Initialize ttnn device
    print("Initializing ttnn device...")
    device = ttnn.open_device(device_id=0)

    try:
        # Create our ttnn model
        print("Creating ttnn model...")
        model = LlamaForCausalLM(config, device=device, state_dict=state_dict)
        model.eval()

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids

        print(f"Input tokens: {input_ids}")
        print(f"Generating response...\n")

        # Generate tokens autoregressively
        with torch.no_grad():
            output_ids = input_ids.clone()

            for step in range(10):  # Generate 10 more tokens
                # Forward pass
                outputs = model(output_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # Greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                output_ids = torch.cat([output_ids, next_token], dim=-1)

                # Decode current output
                decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                print(f"Step {step+1}: {decoded}")

        # Final output
        final_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n{'='*60}")
        print(f"Final output: {final_text}")
        print(f"{'='*60}")

    finally:
        # Clean up
        print("\nClosing device...")
        ttnn.close_device(device)

    print("Done!")


if __name__ == "__main__":
    main()
