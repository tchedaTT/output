"""
Rotary Position Embedding (RoPE) for LLaMA model ported to TTNN.
"""

import math
import ttnn


class LlamaRotaryEmbedding:
    def __init__(self, config, torch_params=None, device=None):
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.device = device
        self.rope_init_fn = self._get_rope_init_function(self.rope_type)

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        # Convert inv_freq to ttnn tensor
        self.inv_freq = ttnn.from_torch(
            inv_freq,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.original_inv_freq = self.inv_freq

    def _get_rope_init_function(self, rope_type):
        """Get the appropriate RoPE initialization function."""
        rope_functions = {
            "default": self._compute_default_rope_parameters,
            "llama3": self._compute_llama3_parameters,
        }
        return rope_functions[rope_type]

    def _compute_default_rope_parameters(self, config, device):
        """Compute default RoPE parameters."""
        import torch
        base = config.rope_theta
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = head_dim
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device="cpu", dtype=torch.float) / dim))
        return inv_freq, attention_factor

    def _compute_llama3_parameters(self, config, device):
        """Compute Llama3-specific RoPE parameters."""
        import torch
        inv_freq, attention_factor = self._compute_default_rope_parameters(config, device)
        factor = config.rope_scaling["factor"]
        low_freq_factor = config.rope_scaling["low_freq_factor"]
        high_freq_factor = config.rope_scaling["high_freq_factor"]
        old_context_len = config.rope_scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama, attention_factor

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        # Split tensor in half along last dimension
        shape = x.shape
        half_dim = shape[-1] // 2

        # Get first and second halves
        x1 = ttnn.slice(x, (0,) * (len(shape) - 1) + (0,),
                       shape[:-1] + (half_dim,))
        x2 = ttnn.slice(x, (0,) * (len(shape) - 1) + (half_dim,),
                       shape)

        # Negate second half and concatenate (-x2, x1)
        neg_x2 = ttnn.neg(x2)
        return ttnn.concat([neg_x2, x1], dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`ttnn.Tensor`): The query tensor.
            k (`ttnn.Tensor`): The key tensor.
            cos (`ttnn.Tensor`): The cosine part of the rotary embedding.
            sin (`ttnn.Tensor`): The sine part of the rotary embedding.
            position_ids (`ttnn.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(ttnn.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = ttnn.unsqueeze(cos, dim=unsqueeze_dim)
        sin = ttnn.unsqueeze(sin, dim=unsqueeze_dim)

        # q_embed = (q * cos) + (rotate_half(q) * sin)
        q_cos = ttnn.multiply(q, cos)
        q_rotated = LlamaRotaryEmbedding.rotate_half(q)
        q_sin = ttnn.multiply(q_rotated, sin)
        q_embed = ttnn.add(q_cos, q_sin)

        # k_embed = (k * cos) + (rotate_half(k) * sin)
        k_cos = ttnn.multiply(k, cos)
        k_rotated = LlamaRotaryEmbedding.rotate_half(k)
        k_sin = ttnn.multiply(k_rotated, sin)
        k_embed = ttnn.add(k_cos, k_sin)

        return q_embed, k_embed

    def __call__(self, x, position_ids):
        """Forward pass - expects x and position_ids to be ttnn tensors on device."""
        # Expand inv_freq to match batch dimension
        batch_size = position_ids.shape[0]
        inv_freq_shape = (batch_size, self.inv_freq.shape[0], 1)
        inv_freq_expanded = ttnn.unsqueeze(self.inv_freq, dim=0)
        inv_freq_expanded = ttnn.unsqueeze(inv_freq_expanded, dim=-1)

        # Expand position_ids
        position_ids_expanded = ttnn.unsqueeze(position_ids, dim=1)

        # Compute frequencies: inv_freq_expanded @ position_ids_expanded
        freqs = ttnn.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = ttnn.transpose(freqs, 1, 2)

        # Concatenate frequencies with themselves
        emb = ttnn.concat([freqs, freqs], dim=-1)

        # Compute cos and sin with attention scaling
        cos = ttnn.cos(emb)
        cos = ttnn.multiply(cos, self.attention_scaling)

        sin = ttnn.sin(emb)
        sin = ttnn.multiply(sin, self.attention_scaling)

        return cos, sin