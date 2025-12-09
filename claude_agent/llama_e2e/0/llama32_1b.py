"""
Simple Llama 3.2 1B implementation in ttnn.

This module provides a clean, educational implementation of Llama 3.2 1B using ttnn,
designed to be compatible with HuggingFace's generate function.
"""

import torch
import torch.nn as nn
import ttnn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple
import argparse


class TtnnRMSNorm(nn.Module):
    """RMSNorm implementation in ttnn."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.device = device

    def forward(self, hidden_states):
        # Convert to ttnn if needed
        is_torch = isinstance(hidden_states, torch.Tensor)

        if is_torch:
            input_dtype = hidden_states.dtype
            hidden_states = ttnn.from_torch(hidden_states, device=self.device)

        # Compute RMS norm: x * rsqrt(mean(x^2) + eps) * weight
        variance = ttnn.pow(hidden_states, 2)
        variance = ttnn.mean(variance, dim=-1, keepdim=True)
        hidden_states = hidden_states * ttnn.rsqrt(variance + self.eps)

        # Apply weight
        weight_tt = ttnn.from_torch(self.weight.to(input_dtype if is_torch else torch.float32), device=self.device)
        hidden_states = ttnn.mul(hidden_states, weight_tt)

        if is_torch:
            hidden_states = ttnn.to_torch(hidden_states)

        return hidden_states


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.
    Uses HuggingFace format (not Meta interleaved format).

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine values [batch, seq_len, head_dim]
        sin: Sine values [batch, seq_len, head_dim]
        position_ids: Position indices [batch, seq_len]
    """
    # Gather cos and sin based on position_ids
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, head_dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]

    # Apply rotary embeddings (HuggingFace format)
    # Split into first and second half
    q_half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :q_half_dim], q[..., q_half_dim:]
    k1, k2 = k[..., :q_half_dim], k[..., q_half_dim:]

    cos = cos[..., :q_half_dim]
    sin = sin[..., :q_half_dim]

    # Rotate: [q1, q2] -> [q1*cos - q2*sin, q1*sin + q2*cos]
    q_embed = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)

    k_embed = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)

    return q_embed, k_embed


class TtnnLlamaAttention(nn.Module):
    """Llama attention module using ttnn operations."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Initialize rope embeddings
        self._init_rope()

    def _init_rope(self):
        """Initialize RoPE embeddings."""
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_rope_cache(self, seq_len, device, dtype):
        """Get cached RoPE embeddings."""
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        # QKV projections using ttnn
        hidden_states_tt = ttnn.from_torch(hidden_states, device=self.device)

        # Note: ttnn.linear expects weights as (in_features, out_features)
        # while PyTorch uses (out_features, in_features), so we transpose
        q_weight_tt = ttnn.from_torch(self.q_proj.weight.T, device=self.device)
        k_weight_tt = ttnn.from_torch(self.k_proj.weight.T, device=self.device)
        v_weight_tt = ttnn.from_torch(self.v_proj.weight.T, device=self.device)

        query_states_tt = ttnn.linear(hidden_states_tt, q_weight_tt)
        key_states_tt = ttnn.linear(hidden_states_tt, k_weight_tt)
        value_states_tt = ttnn.linear(hidden_states_tt, v_weight_tt)

        # Convert back to torch for reshaping
        query_states = ttnn.to_torch(query_states_tt)
        key_states = ttnn.to_torch(key_states_tt)
        value_states = ttnn.to_torch(value_states_tt)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get RoPE embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self._get_rope_cache(kv_seq_len, hidden_states.device, hidden_states.dtype)

        # Apply RoPE
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None

        # Repeat KV heads for grouped-query attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Convert to ttnn for attention computation
        query_states_tt = ttnn.from_torch(query_states, device=self.device)
        key_states_tt = ttnn.from_torch(key_states, device=self.device)
        value_states_tt = ttnn.from_torch(value_states, device=self.device)

        # Use ttnn scaled dot product attention
        if attention_mask is not None:
            attention_mask_tt = ttnn.from_torch(attention_mask, device=self.device)
        else:
            attention_mask_tt = None

        attn_output_tt = ttnn.transformer.scaled_dot_product_attention(
            query_states_tt,
            key_states_tt,
            value_states_tt,
            attn_mask=attention_mask_tt,
            is_causal=(attention_mask is None),
        )

        attn_output = ttnn.to_torch(attn_output_tt)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        attn_output_tt = ttnn.from_torch(attn_output, device=self.device)
        o_weight_tt = ttnn.from_torch(self.o_proj.weight.T, device=self.device)
        attn_output_tt = ttnn.linear(attn_output_tt, o_weight_tt)
        attn_output = ttnn.to_torch(attn_output_tt)

        return attn_output, past_key_value


class TtnnLlamaMLP(nn.Module):
    """Llama MLP module using ttnn operations."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Convert to ttnn
        x_tt = ttnn.from_torch(x, device=self.device)

        # Gate projection (for SwiGLU activation)
        gate_weight_tt = ttnn.from_torch(self.gate_proj.weight.T, device=self.device)
        gate = ttnn.linear(x_tt, gate_weight_tt)
        gate = ttnn.silu(gate)  # SiLU activation

        # Up projection
        up_weight_tt = ttnn.from_torch(self.up_proj.weight.T, device=self.device)
        up = ttnn.linear(x_tt, up_weight_tt)

        # Element-wise multiplication (SwiGLU)
        intermediate = ttnn.mul(gate, up)

        # Down projection
        down_weight_tt = ttnn.from_torch(self.down_proj.weight.T, device=self.device)
        output = ttnn.linear(intermediate, down_weight_tt)

        return ttnn.to_torch(output)


class TtnnLlamaDecoderLayer(nn.Module):
    """Llama decoder layer using ttnn operations."""

    def __init__(self, config, device=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = TtnnLlamaAttention(config, device=device)
        self.mlp = TtnnLlamaMLP(config, device=device)
        self.input_layernorm = TtnnRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.post_attention_layernorm = TtnnRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states

        # Self attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class TtnnLlamaModel(nn.Module):
    """Main Llama model using ttnn operations."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            TtnnLlamaDecoderLayer(config, device=device) for _ in range(config.num_hidden_layers)
        ])
        self.norm = TtnnRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                # Decode: position_ids are offset by cache length
                past_length = past_key_values[0][0].shape[2]
                position_ids = torch.arange(past_length, past_length + seq_length, dtype=torch.long, device=input_ids.device)
            else:
                # Prefill: position_ids are 0 to seq_length-1
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Prepare attention mask
        if attention_mask is not None:
            # Convert to 4D mask [batch, 1, seq_len, kv_seq_len]
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, past_key_values
            )

        # Forward through layers
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # Return in HuggingFace format
        from transformers.modeling_outputs import BaseModelOutputWithPast
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=None,
            attentions=None,
        )

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, hidden_states, past_key_values):
        """Prepare 4D attention mask."""
        # Create causal mask
        batch_size, seq_length = input_shape
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

        # Create causal mask [batch, 1, seq_len, kv_seq_len]
        combined_attention_mask = None
        if seq_length > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape, hidden_states.dtype, device=hidden_states.device, past_length=past_length
            )

        if attention_mask is not None:
            # Expand attention mask [batch, seq_len] -> [batch, 1, seq_len, kv_seq_len]
            expanded_attn_mask = self._expand_mask(attention_mask, hidden_states.dtype, tgt_len=seq_length)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @staticmethod
    def _make_causal_mask(input_shape, dtype, device, past_length=0):
        """Make causal mask for attention."""
        batch_size, target_length = input_shape
        mask = torch.full((target_length, target_length), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_length > 0:
            mask = torch.cat([torch.zeros(target_length, past_length, dtype=dtype, device=device), mask], dim=-1)

        return mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_length)

    @staticmethod
    def _expand_mask(mask, dtype, tgt_len=None):
        """Expand attention mask."""
        batch_size, src_length = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_length

        expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_length).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class TtnnLlamaForCausalLM(nn.Module):
    """Llama for causal language modeling, compatible with HuggingFace generate()."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.model = TtnnLlamaModel(config, device=device)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Needed for HuggingFace compatibility
        self.config.is_encoder_decoder = False
        self.config.model_type = "llama"
        self.main_input_name = "input_ids"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state

        # LM head
        hidden_states_tt = ttnn.from_torch(hidden_states, device=self.device)
        lm_head_weight_tt = ttnn.from_torch(self.lm_head.weight.T, device=self.device)
        logits_tt = ttnn.linear(hidden_states_tt, lm_head_weight_tt)
        logits = ttnn.to_torch(logits_tt)

        loss = None
        if labels is not None:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation (required by HuggingFace generate)."""
        if past_key_values:
            # Only use last token if we have past_key_values
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search (required by HuggingFace generate)."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    def can_generate(self):
        """Check if model can generate (required by HuggingFace)."""
        return True

    @classmethod
    def from_pretrained(cls, model_name, device=None):
        """Load pretrained model from HuggingFace."""
        # Load config and reference model
        config = AutoConfig.from_pretrained(model_name)
        reference_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Create ttnn model
        ttnn_model = cls(config, device=device)

        # Copy weights from reference model
        ttnn_model.load_state_dict(reference_model.state_dict(), strict=False)

        return ttnn_model


def main():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 1B with ttnn")
    parser.add_argument("--prompt", type=str, default="1 2 3 4 5 6 7 8 9 10 11 12",
                        help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model name on HuggingFace")
    args = parser.parse_args()

    # Initialize ttnn device
    device = ttnn.open_device(device_id=0)

    print("Loading model and tokenizer...")
    model = TtnnLlamaForCausalLM.from_pretrained(args.model_name, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    print(f"\nPrompt: {args.prompt}")
    print("Generating...")

    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt")

    # Generate using HuggingFace's generate function
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")

    # Close device
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
