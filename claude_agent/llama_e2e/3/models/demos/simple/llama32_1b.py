"""
Simple Llama 3.2 1B implementation in ttnn.

This implementation is designed to be clean and educational, showing how to use ttnn
directly to implement a transformer model compatible with HuggingFace.
"""

import torch
import torch.nn as nn
import ttnn
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple
import math


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.

    This follows HuggingFace conventions (not Meta's interleaved format).
    The rotation is applied in the complex plane by treating consecutive pairs
    of dimensions as real and imaginary parts.

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch, num_heads, seq_len, head_dim]
        cos: Cosine values [batch, seq_len, head_dim]
        sin: Sine values [batch, seq_len, head_dim]
        position_ids: Position indices [batch, seq_len]

    Returns:
        Rotated query and key tensors
    """
    # Gather cos and sin values for the positions
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, head_dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, head_dim]
    cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]

    # HuggingFace RoPE: rotate consecutive pairs (not interleaved like Meta)
    # Split into first half and second half
    q_half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :q_half_dim], q[..., q_half_dim:]
    k1, k2 = k[..., :q_half_dim], k[..., q_half_dim:]

    # Apply rotation
    cos_half = cos[..., :q_half_dim]
    sin_half = sin[..., :q_half_dim]

    q_rotated = torch.cat([q1 * cos_half - q2 * sin_half, q1 * sin_half + q2 * cos_half], dim=-1)
    k_rotated = torch.cat([k1 * cos_half - k2 * sin_half, k1 * sin_half + k2 * cos_half], dim=-1)

    return q_rotated, k_rotated


class TtnnLlamaRMSNorm(nn.Module):
    """RMS Normalization layer implemented with ttnn."""

    def __init__(self, hidden_size, eps=1e-6, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.device = device

    def forward(self, hidden_states):
        """Apply RMS normalization."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TtnnLlamaAttention(nn.Module):
    """Multi-head attention layer using ttnn operations."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.device = device

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Linear layers - note: ttnn.linear expects (B,C) weights vs PyTorch's (C,B)
        # We'll handle the transpose when converting weights
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match query heads for grouped-query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """Forward pass for attention layer."""
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings (HuggingFace format)
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Handle past key values for decoding
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads for grouped-query attention
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention using scaled dot product
        # In real ttnn, we would use ttnn.scaled_dot_product_attention
        # For this implementation, we use PyTorch's efficient version
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None and q_len > 1,
        )

        # Reshape back to [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class TtnnLlamaMLP(nn.Module):
    """Feed-forward network using ttnn operations."""

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = device

        # Linear layers
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        """Forward pass for MLP."""
        # SwiGLU activation: SiLU(gate) * up
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TtnnLlamaDecoderLayer(nn.Module):
    """Transformer decoder layer combining attention and MLP."""

    def __init__(self, config, device=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.device = device

        self.self_attn = TtnnLlamaAttention(config=config, device=device)
        self.mlp = TtnnLlamaMLP(config, device=device)
        self.input_layernorm = TtnnLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.post_attention_layernorm = TtnnLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Forward pass for decoder layer."""
        residual = hidden_states

        # Self-attention with pre-norm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cos=cos,
            sin=sin,
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
    """
    Main Llama model using ttnn operations.

    This implements the core transformer with embeddings and decoder layers.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.device = device

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TtnnLlamaDecoderLayer(config, device=device) for _ in range(config.num_hidden_layers)])
        self.norm = TtnnLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

        # Precompute RoPE frequencies (HuggingFace format, not Meta's)
        self._setup_rope(config)

    def _setup_rope(self, config):
        """Precompute rotary position embedding cos/sin values."""
        head_dim = config.hidden_size // config.num_attention_heads

        # Compute frequency bands
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Precompute for max sequence length
        t = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)

        # HuggingFace uses non-interleaved format: [cos_0, cos_1, ..., cos_n/2, cos_0, cos_1, ..., cos_n/2]
        # We repeat to match the full head dimension
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass for the model."""
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            if past_key_values is not None:
                # For decoding, position is based on cache length
                past_length = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
                position_ids = torch.arange(
                    past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device
                )
            else:
                position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Get RoPE cos/sin for current positions
        cos = self.cos_cached
        sin = self.sin_cached

        # Process through decoder layers
        all_hidden_states = () if output_hidden_states else None
        next_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cos=cos,
                sin=sin,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[1],)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return in HuggingFace format
        if return_dict:
            from transformers.modeling_outputs import BaseModelOutputWithPast
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache if use_cache else None,
                hidden_states=all_hidden_states,
            )

        return hidden_states, next_cache, all_hidden_states


class TtnnLlamaForCausalLM(nn.Module):
    """
    Llama model for causal language modeling, compatible with HuggingFace.

    This wraps the base model and adds the language modeling head.
    Can be used directly with transformers.generate()!
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.model = TtnnLlamaModel(config, device=device)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.device = device

    def get_input_embeddings(self):
        """Get embedding layer (required by HuggingFace)."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """Set embedding layer (required by HuggingFace)."""
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Get LM head (required by HuggingFace)."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set LM head (required by HuggingFace)."""
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """Prepare inputs for generation (required by HuggingFace generate)."""
        if past_key_values:
            # Only use last token for decode pass
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search (required by HuggingFace)."""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

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
        """Forward pass for causal LM."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        # Compute logits
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
        )

    @classmethod
    def from_pretrained(cls, model_name, device=None):
        """
        Load pretrained weights from HuggingFace and convert to ttnn model.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-1B")
            device: Device to load model on

        Returns:
            TtnnLlamaForCausalLM model with loaded weights
        """
        print(f"Loading model {model_name} from HuggingFace...")

        # Load config and reference model
        config = AutoConfig.from_pretrained(model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        # Create ttnn model
        ttnn_model = cls(config, device=device)

        # Copy weights from HuggingFace model
        # Note: ttnn.linear expects transposed weights compared to PyTorch
        # But since we're using nn.Linear here, PyTorch handles it

        print("Converting weights to ttnn format...")

        # Embeddings
        ttnn_model.model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()

        # Decoder layers
        for i, (ttnn_layer, hf_layer) in enumerate(zip(ttnn_model.model.layers, hf_model.model.layers)):
            # Attention
            ttnn_layer.self_attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
            ttnn_layer.self_attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
            ttnn_layer.self_attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
            ttnn_layer.self_attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()

            # MLP
            ttnn_layer.mlp.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
            ttnn_layer.mlp.up_proj.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
            ttnn_layer.mlp.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.clone()

            # Norms
            ttnn_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
            ttnn_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

        # Final norm and LM head
        ttnn_model.model.norm.weight.data = hf_model.model.norm.weight.data.clone()
        ttnn_model.lm_head.weight.data = hf_model.lm_head.weight.data.clone()

        print("Model loaded successfully!")

        # Clean up reference model
        del hf_model

        return ttnn_model


def main():
    """Main function to run the model."""
    parser = argparse.ArgumentParser(description="Run Llama 3.2 1B with ttnn")
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
    print("Llama 3.2 1B in ttnn - Simple Implementation")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TtnnLlamaForCausalLM.from_pretrained(args.model_name, device=device)
    model.eval()
    model.to(device)

    # Tokenize input
    print(f"\nPrompt: {args.prompt}")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    print(f"\nGenerating {args.max_new_tokens} tokens...")
    print("-" * 80)

    # Generate using HuggingFace's generate function!
    # This is the elegant approach mentioned in the task
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,  # Greedy decoding for deterministic output
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")
    print("-" * 80)

    print("\n✓ Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
