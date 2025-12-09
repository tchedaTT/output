# MODEL_BRINGUP.md - Llama 3.2 1B in ttnn

## Overview

This document captures learnings, gotchas, and best practices discovered while implementing Llama 3.2 1B in ttnn. The goal was to create a clean, educational implementation that demonstrates how to use ttnn directly while maintaining compatibility with HuggingFace's ecosystem.

## Key Learnings

### 1. ttnn vs PyTorch Weight Format

**Critical Difference**: `ttnn.linear` expects weights in shape `(B, C)` while PyTorch's `torch.nn.functional.linear` expects `(C, B)`.

- **PyTorch**: `F.linear(input, weight)` with input `(A, B)` and weight `(C, B)` → output `(A, C)`
- **ttnn**: `ttnn.linear(input, weight)` with input `(A, B)` and weight `(B, C)` → output `(A, C)`

**Implication**: When converting weights from HuggingFace models, you need to transpose weight matrices if using native ttnn.linear. In this implementation, we used `nn.Linear` which handles this internally, but in pure ttnn implementations, remember to transpose!

```python
# If using pure ttnn.linear:
hf_weight = hf_model.layer.weight  # Shape: (out_features, in_features)
ttnn_weight = hf_weight.T  # Transpose to (in_features, out_features)
```

### 2. RoPE Format - HuggingFace vs Meta

**Major Gotcha**: The Rotary Position Embedding (RoPE) format differs between HuggingFace and Meta implementations!

- **Meta format**: Interleaved - rotates pairs like `[x0, x1, x2, x3, ...]` → `[rotate(x0,x1), rotate(x2,x3), ...]`
- **HuggingFace format**: Split into halves - `[x0, x1, ..., x_{n/2}]` and `[x_{n/2+1}, ..., x_n]` treated as separate chunks

**Why This Matters**: The tt_transformers library swizzles weights to convert HuggingFace format to Meta format. Since we're loading a HuggingFace model (even though it's from Meta), we must:
- **NOT** use the rotary_embedding ops from tt_transformers' Llama implementation
- Keep the HuggingFace convention throughout
- Split head dimensions in half, not interleave them

**Implementation**:
```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Split into halves (HuggingFace style)
    q_half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :q_half_dim], q[..., q_half_dim:]
    k1, k2 = k[..., :q_half_dim], k[..., q_half_dim:]

    # Apply rotation
    cos_half = cos[..., :q_half_dim]
    sin_half = sin[..., :q_half_dim]

    q_rotated = torch.cat([q1 * cos_half - q2 * sin_half,
                           q1 * sin_half + q2 * cos_half], dim=-1)
    k_rotated = torch.cat([k1 * cos_half - k2 * sin_half,
                           k1 * sin_half + k2 * cos_half], dim=-1)
    return q_rotated, k_rotated
```

### 3. HuggingFace Compatibility Pattern

**Best Practice**: Making your ttnn model compatible with HuggingFace's `generate()` function is much cleaner than writing custom generation logic!

**Required Methods**:
- `get_input_embeddings()` / `set_input_embeddings()` - For embedding access
- `get_output_embeddings()` / `set_output_embeddings()` - For LM head access
- `prepare_inputs_for_generation()` - Handles prefill vs decode pass logic
- `_reorder_cache()` - For beam search support
- Return `CausalLMOutputWithPast` or compatible format from `forward()`

**Benefits**:
- Automatic KV cache management
- Support for various decoding strategies (greedy, beam search, sampling)
- Integration with HuggingFace ecosystem
- Less code to maintain

### 4. Model Architecture Details - Llama 3.2 1B

Key architectural features:
- **Grouped-Query Attention (GQA)**: Fewer key/value heads than query heads
  - `num_attention_heads` = query heads
  - `num_key_value_heads` = key/value heads (typically fewer)
  - Requires repeating K/V heads to match Q heads
- **SwiGLU Activation**: `SiLU(gate_proj(x)) * up_proj(x)`
- **RMSNorm**: More efficient than LayerNorm, no mean centering
- **Pre-normalization**: Norm before attention/MLP, not after

### 5. Prefill vs Decode Passes

**Prefill** (first forward pass):
- Processes entire prompt at once
- Full attention matrix computed
- KV cache is empty, gets filled
- `past_key_values = None`

**Decode** (subsequent passes):
- Processes one token at a time
- Only computes attention for new token against all previous
- Reuses KV cache from previous tokens
- `past_key_values` contains cached keys/values
- Input is only the last token: `input_ids = input_ids[:, -1:]`

**Implementation Detail**: The `prepare_inputs_for_generation()` method handles this automatically:
```python
if past_key_values:
    input_ids = input_ids[:, -1:]  # Only last token for decode
```

### 6. ttnn Functions to Use

Based on exploring tt_transformers, here are key ttnn functions for transformer models:

- **`ttnn.scaled_dot_product_attention`**: Use this instead of manual attention! Handles:
  - Scaling by `1/sqrt(d_k)`
  - Attention mask application
  - Causal masking
  - Efficient computation

- **`ttnn.linear`**: Matrix multiplication with transposed weight convention

- **`ttnn.embedding`**: Token embedding lookup

- **RMSNorm operations**: Can be implemented with ttnn primitives:
  - `ttnn.pow()` for squaring
  - `ttnn.mean()` for variance
  - `ttnn.rsqrt()` for inverse square root

### 7. Weight Loading Strategy

**Approach**: Load HuggingFace model first, then copy weights:

```python
@classmethod
def from_pretrained(cls, model_name, device=None):
    config = AutoConfig.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    ttnn_model = cls(config, device=device)

    # Copy weights layer by layer
    # Be careful with transpositions if using pure ttnn!

    return ttnn_model
```

**Why**: Easier than manual weight download/conversion, leverages HuggingFace's model hub infrastructure.

### 8. Debugging Strategies

**Recommended Approach**: Binary search / bisection method

When correctness issues arise:
1. Create wrapper functions that convert ttnn → torch, use HF module, convert back
2. Swap out components one at a time to isolate the bug
3. Example: If attention output is wrong, temporarily use HF's attention module
4. Much faster than layer-by-layer PCC checking for simple models

**Code Pattern**:
```python
# Debugging wrapper example
def debug_attention(hidden_states, ...):
    # Convert to torch
    torch_hidden = ttnn_to_torch(hidden_states)

    # Use reference HF implementation
    output = hf_model.layers[i].self_attn(torch_hidden, ...)

    # Convert back
    return torch_to_ttnn(output)
```

### 9. Common Pitfalls

1. **Forgetting to handle batch dimensions**: ttnn operations may have different broadcasting rules
2. **Position IDs**: Must adjust for cached sequence length during decode
3. **Attention mask shape**: Causal mask can be implicit or explicit
4. **Device placement**: Ensure all tensors are on the same device
5. **Data types**: Be consistent with float32/float16/bfloat16

### 10. Testing Best Practices

**Test Prompt**: `"1 2 3 4 5 6 7 8 9 10 11 12"`
- Simple pattern the model should easily continue
- Easy to verify correctness (should continue: 13 14 15...)
- Tests both understanding and generation

**Validation Strategy**:
1. First test with reference HuggingFace model to verify expected output
2. Compare ttnn model output token-by-token
3. Check both prefill and decode passes work correctly
4. Verify KV cache is properly maintained across decode steps

### 11. Code Organization

**Single-File Approach**:
- All classes in one file for simplicity
- Clear separation: RMSNorm → MLP → Attention → DecoderLayer → Model → CausalLM
- Bottom-up dependency structure (no circular imports)

**Naming Convention**:
- Prefix with `TtnnLlama` to distinguish from HF classes
- Keep method names matching HuggingFace for compatibility

### 12. Performance Considerations (Future Work)

For production use, consider:
- **Multi-device**: Shard layers across devices
- **Weight quantization**: INT8/INT4 for efficiency
- **Flash Attention**: More memory-efficient attention computation
- **Fused kernels**: Combine operations to reduce memory transfers
- **KV cache optimization**: Quantized cache, paged attention

## Summary - What Would Have Saved Time

1. **Understanding ttnn.linear weight format difference** - Document this prominently!
2. **RoPE format mismatch** - Biggest gotcha, could waste hours debugging
3. **HuggingFace compatibility pattern** - Way easier than custom generation
4. **Using scaled_dot_product_attention** - Don't implement attention manually
5. **Binary search debugging** - More efficient than exhaustive PCC checking

## Running the Model

```bash
# Basic usage with default prompt
python models/demos/simple/llama32_1b.py

# Custom prompt
python models/demos/simple/llama32_1b.py --prompt "Hello, how are you?"

# Generate more tokens
python models/demos/simple/llama32_1b.py --max-new-tokens 50
```

## Device Management

If the Wormhole device gets into a bad state:
```bash
tt-smi -r  # Reset the device
```

## Conclusion

This implementation demonstrates how to bring up a transformer model in ttnn while maintaining simplicity and HuggingFace compatibility. The key is understanding the framework differences (weight formats, RoPE conventions) and leveraging existing tools (HuggingFace generate) rather than reinventing the wheel.

The most valuable lesson: **Take time to understand the conventions before diving into implementation.** Reading through existing code (tt_transformers) for inspiration and understanding available functions is time well spent!
