# Llama 3.2 1B Model Bringup Guide

This document captures key learnings, gotchas, and best practices discovered while implementing Llama 3.2 1B in ttnn from scratch.

## Table of Contents
1. [Overview](#overview)
2. [Key ttnn Concepts](#key-ttnn-concepts)
3. [Weight Conversion](#weight-conversion)
4. [Critical Gotchas](#critical-gotchas)
5. [Implementation Approach](#implementation-approach)
6. [Debugging Strategies](#debugging-strategies)
7. [HuggingFace Integration](#huggingface-integration)
8. [Performance Considerations](#performance-considerations)

---

## Overview

The goal of this implementation is to create a clean, educational example of porting a HuggingFace model (Llama 3.2 1B) to run on ttnn hardware accelerators. The implementation focuses on:

- **Simplicity**: Direct ttnn operations without complex abstractions
- **Correctness**: Faithful reproduction of the original model behavior
- **Compatibility**: Works with HuggingFace's `generate()` function
- **Educational value**: Shows developers how to use ttnn directly

---

## Key ttnn Concepts

### 1. Weight Transposition for Linear Layers

**Most important gotcha**: `ttnn.linear` expects different weight dimensions than PyTorch!

- **PyTorch**: `torch.nn.functional.linear(input, weight)` expects:
  - Input: `[..., in_features]` shape `(A, B)`
  - Weight: `[out_features, in_features]` shape `(C, B)`
  - Output: `[..., out_features]` shape `(A, C)`
  - Math: `input @ weight.T`

- **ttnn**: `ttnn.linear(input, weight)` expects:
  - Input: `[..., in_features]` shape `(A, B)`
  - Weight: `[in_features, out_features]` shape `(B, C)` **(transposed!)**
  - Output: `[..., out_features]` shape `(A, C)`
  - Math: `input @ weight`

**Action**: Always transpose PyTorch linear weights before converting to ttnn:
```python
# PyTorch weight shape: [out_features, in_features]
hf_weight = hf_layer.linear.weight

# Transpose for ttnn
ttnn_weight = hf_weight.T  # Now [in_features, out_features]

# Convert to ttnn
ttnn_weight_tensor = ttnn.from_torch(
    ttnn_weight,
    device=device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)
```

### 2. Tensor Layouts

ttnn supports different memory layouts optimized for different operations:

- **TILE_LAYOUT**: Default layout for compute operations, optimized for matrix operations
- **ROW_MAJOR_LAYOUT**: Standard row-major layout, useful for some operations

For most operations, use `TILE_LAYOUT`:
```python
tensor = ttnn.from_torch(
    torch_tensor,
    device=device,
    layout=ttnn.TILE_LAYOUT,  # Use TILE_LAYOUT for compute
    memory_config=ttnn.DRAM_MEMORY_CONFIG  # DRAM interleaved is fine for simple models
)
```

### 3. Memory Configuration

For a simple single-device model, `DRAM_MEMORY_CONFIG` (DRAM interleaved) is sufficient:
```python
ttnn.DRAM_MEMORY_CONFIG  # Use this for all tensors in simple implementations
```

Later optimizations can explore:
- L1 memory for frequently accessed tensors
- Sharded memory for multi-device setups
- Custom memory configurations

### 4. Device Management

```python
# Open device
device = ttnn.open_device(device_id=0)

# ... use device ...

# Always close when done
ttnn.close_device(device)
```

If your device gets into a bad state:
```bash
tt-smi -r  # Reset the device
```

---

## Weight Conversion

### Converting HuggingFace Weights to ttnn

The general pattern for converting weights:

```python
# 1. Extract PyTorch weight
torch_weight = hf_model.layer.weight

# 2. Transpose if it's a linear layer weight
if is_linear_weight:
    torch_weight = torch_weight.T

# 3. Convert to ttnn
ttnn_weight = ttnn.from_torch(
    torch_weight,
    device=device,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)
```

### Embeddings

Embedding weights can be used directly (no transpose needed):
```python
embed_weight = hf_model.embed_tokens.weight
ttnn_embed = ttnn.from_torch(embed_weight, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Usage
hidden_states = ttnn.embedding(input_ids_ttnn, ttnn_embed)
```

---

## Critical Gotchas

### 1. RoPE Format: HuggingFace vs Meta

**Critical distinction**: The Llama 3.2 model from HuggingFace uses HuggingFace's RoPE format, NOT Meta's format!

- **Meta format**: Interleaved - requires weight swizzling
- **HuggingFace format**: Separate cos/sin coefficients - no swizzling needed

**Do NOT use** the RoPE functions from `tt_transformers` that expect Meta format. They will swizzle weights incorrectly.

**Correct approach for HuggingFace models**:
```python
def apply_rotary_emb_huggingface_format(x, freqs_cos, freqs_sin):
    # Split features in half
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Rotate: [x1*cos - x2*sin, x1*sin + x2*cos]
    out1 = ttnn.subtract(ttnn.multiply(x1, freqs_cos), ttnn.multiply(x2, freqs_sin))
    out2 = ttnn.add(ttnn.multiply(x1, freqs_sin), ttnn.multiply(x2, freqs_cos))

    return ttnn.concat([out1, out2], dim=-1)
```

Pre-compute cos/sin frequencies:
```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
t = torch.arange(max_seq_len, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)  # Duplicate for HuggingFace format

cos_cached = ttnn.from_torch(emb.cos()[None, None, :, :], device=device, ...)
sin_cached = ttnn.from_torch(emb.sin()[None, None, :, :], device=device, ...)
```

### 2. RMSNorm Implementation

RMSNorm is used instead of LayerNorm in Llama models:

```python
def rmsnorm(x, weight, eps=1e-6):
    # Calculate variance (mean of squares)
    variance = ttnn.mean(ttnn.pow(x, 2.0), dim=-1, keepdim=True)

    # Normalize: x / sqrt(variance + eps)
    x_normed = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(variance, eps)))

    # Apply learned weight
    return ttnn.multiply(x_normed, weight)
```

### 3. SwiGLU Activation

Llama uses SwiGLU activation in the MLP, not standard ReLU:

```python
def swiglu_mlp(x, gate_proj, up_proj, down_proj):
    gate = ttnn.linear(x, gate_proj)
    up = ttnn.linear(x, up_proj)

    # SwiGLU: silu(gate) * up
    gate = ttnn.silu(gate)
    hidden = ttnn.multiply(gate, up)

    # Project down
    output = ttnn.linear(hidden, down_proj)
    return output
```

### 4. Grouped-Query Attention (GQA)

Llama 3.2 1B uses grouped-query attention where K/V heads are fewer than Q heads:

```python
num_key_value_groups = num_heads // num_key_value_heads

if num_key_value_groups > 1:
    # Repeat K/V to match Q head count
    key_states = ttnn.repeat_interleave(key_states, num_key_value_groups, dim=2)
    value_states = ttnn.repeat_interleave(value_states, num_key_value_groups, dim=2)
```

### 5. Attention Implementation

Use ttnn's built-in scaled dot-product attention instead of implementing manually:

```python
attn_output = ttnn.transformer.scaled_dot_product_attention(
    query_states,  # [bsz, n_heads, seq_len, head_dim]
    key_states,    # [bsz, n_heads, seq_len, head_dim]
    value_states,  # [bsz, n_heads, seq_len, head_dim]
    is_causal=True,
    attention_mask=attention_mask
)
```

This is more efficient and handles the scaling factor automatically.

---

## Implementation Approach

### Module Structure

Organize the code into clean, reusable modules:

1. **LlamaRMSNorm**: Normalization layer
2. **LlamaAttention**: Multi-head attention with RoPE
3. **LlamaMLP**: Feed-forward network with SwiGLU
4. **LlamaDecoderLayer**: Single transformer layer
5. **TtnnLlamaForCausalLM**: Complete model wrapper

### Layer-by-Layer Construction

Build the model bottom-up:

1. Start with basic operations (RMSNorm, linear layers)
2. Implement attention mechanism (most complex part)
3. Add MLP
4. Combine into decoder layer
5. Stack layers and add embeddings/head

### Residual Connections

Llama uses pre-norm with residual connections:

```python
# Attention block
residual = hidden_states
hidden_states = layernorm(hidden_states)
hidden_states = attention(hidden_states)
hidden_states = ttnn.add(residual, hidden_states)

# MLP block
residual = hidden_states
hidden_states = layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = ttnn.add(residual, hidden_states)
```

---

## Debugging Strategies

### 1. Bisection with Reference Implementation

When output is incorrect, bisect to find the problematic layer:

```python
# Create a hybrid layer that uses HuggingFace for some parts
def debug_layer(hidden_states_ttnn):
    # Convert to torch
    hidden_torch = ttnn.to_torch(hidden_states_ttnn)

    # Use HuggingFace layer
    output_torch = hf_model.layers[i](hidden_torch)

    # Convert back to ttnn
    output_ttnn = ttnn.from_torch(output_torch, device=device, ...)

    return output_ttnn
```

Replace ttnn layers one by one with reference implementations until you identify the issue. This is faster than checking PCC layer by layer.

### 2. Shape Debugging

Print shapes liberally during development:

```python
print(f"Input shape: {hidden_states.shape}")
print(f"After projection: {query_states.shape}")
print(f"After reshape: {query_states.shape}")
```

Shape mismatches are common and easy to fix once identified.

### 3. Weight Inspection

Verify weights were converted correctly:

```python
# Check a weight after conversion
ttnn_weight_back = ttnn.to_torch(ttnn_weight)
torch_weight_transposed = hf_weight.T

print(f"Weights match: {torch.allclose(ttnn_weight_back, torch_weight_transposed)}")
```

### 4. Numerical Comparison

Compare intermediate outputs:

```python
# ttnn forward
ttnn_output = ttnn_model(input_ids)

# Reference forward
with torch.no_grad():
    hf_output = hf_model(input_ids)

# Compare
diff = (ttnn_output.logits - hf_output.logits).abs().max()
print(f"Max difference: {diff}")
```

---

## HuggingFace Integration

### Making the Model Compatible with `generate()`

The elegant approach is to make your ttnn model work with HuggingFace's `generate()` function:

```python
class TtnnLlamaForCausalLM:
    def forward(self, input_ids, **kwargs):
        # ... ttnn implementation ...
        logits_torch = ttnn.to_torch(logits)

        # Return object with logits attribute
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(logits_torch)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def generate(self, input_ids, **kwargs):
        # Use HuggingFace's GenerationMixin
        from transformers.generation import GenerationMixin

        class GeneratableTtnnLlama(GenerationMixin):
            def __init__(self, ttnn_model):
                self.ttnn_model = ttnn_model
                self.config = ttnn_model.config
                self.generation_config = ttnn_model.generation_config

            def forward(self, *args, **kwargs):
                return self.ttnn_model.forward(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                return {"input_ids": input_ids}

        gen_model = GeneratableTtnnLlama(self)
        return gen_model.generate(input_ids, **kwargs)
```

This allows you to use the model exactly like a HuggingFace model:

```python
model = TtnnLlamaForCausalLM("meta-llama/Llama-3.2-1B")
output_ids = model.generate(input_ids, max_new_tokens=20)
```

### Required Attributes

For HuggingFace compatibility, ensure your model has:
- `config`: Model configuration
- `generation_config`: Generation configuration
- `forward()`: Forward pass returning object with `.logits`
- `__call__()`: Make the model callable
- `prepare_inputs_for_generation()`: Prepare inputs for generation

---

## Performance Considerations

### For This Simple Implementation

The current implementation prioritizes:
1. **Correctness**: Accurate reproduction of model behavior
2. **Simplicity**: Easy to understand and modify
3. **Educational value**: Shows ttnn concepts clearly

### Future Optimizations

Once the model works correctly, consider:

1. **Memory optimization**:
   - Move frequently accessed tensors to L1 memory
   - Use custom memory configurations for different tensors

2. **Multi-device support**:
   - Shard layers across multiple devices
   - Pipeline parallelism for large batches

3. **Operator fusion**:
   - Fuse sequences of operations (e.g., linear + activation)
   - Reduce memory transfers

4. **KV cache**:
   - Cache key/value states during generation
   - Avoid recomputing for previous tokens

5. **Mixed precision**:
   - Use bfloat16 for activations
   - Keep higher precision where needed

### Profiling

Use ttnn profiling tools to identify bottlenecks:
```python
# Profile execution
with ttnn.profile():
    output = model(input_ids)

# Analyze results
ttnn.print_profile()
```

---

## Common ttnn Operations Reference

### Tensor Creation
```python
# From PyTorch
ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# To PyTorch
torch_tensor = ttnn.to_torch(ttnn_tensor)
```

### Basic Operations
```python
# Arithmetic
result = ttnn.add(a, b)
result = ttnn.subtract(a, b)
result = ttnn.multiply(a, b)
result = ttnn.divide(a, b)

# Activation functions
result = ttnn.relu(x)
result = ttnn.silu(x)  # SiLU/Swish
result = ttnn.gelu(x)

# Matrix operations
result = ttnn.linear(input, weight)  # input @ weight
result = ttnn.matmul(a, b)

# Reductions
result = ttnn.mean(x, dim=-1, keepdim=True)
result = ttnn.sum(x, dim=-1, keepdim=True)

# Utilities
result = ttnn.pow(x, 2.0)
result = ttnn.sqrt(x)
result = ttnn.rsqrt(x)  # 1/sqrt(x), more efficient
```

### Shape Operations
```python
# Reshape
result = ttnn.reshape(x, new_shape)

# Permute (transpose)
result = ttnn.permute(x, (0, 2, 1, 3))

# Concatenate
result = ttnn.concat([a, b], dim=-1)

# Repeat
result = ttnn.repeat_interleave(x, repeats, dim=2)
```

### Transformer Operations
```python
# Embeddings
result = ttnn.embedding(input_ids, embedding_weight)

# Attention
result = ttnn.transformer.scaled_dot_product_attention(
    query, key, value,
    is_causal=True,
    attention_mask=mask
)
```

---

## Testing Strategy

### Test Prompt

Use a simple, predictable prompt to verify model behavior:
```python
prompt = "1 2 3 4 5 6 7 8 9 10 11 12"
```

The model should continue the number sequence. This is:
- **Simple**: Easy to verify correctness
- **Deterministic**: Numbers follow a clear pattern
- **Diagnostic**: If the model fails this, something is fundamentally wrong

### Verification Steps

1. **Test with reference model first**:
   ```python
   # Verify the prompt works with HuggingFace model
   hf_output = hf_model.generate(input_ids, max_new_tokens=20)
   print(tokenizer.decode(hf_output[0]))
   ```

2. **Test ttnn model**:
   ```python
   ttnn_output = ttnn_model.generate(input_ids, max_new_tokens=20)
   print(tokenizer.decode(ttnn_output[0]))
   ```

3. **Compare outputs**:
   - Should generate similar continuations
   - Minor differences are acceptable due to numerical precision
   - Major differences indicate a bug

### Correctness Criteria

✅ **Good**: Model continues the sequence reasonably (e.g., "13 14 15...")
✅ **Acceptable**: Model generates coherent text
❌ **Bad**: Model generates gibberish or repeated tokens

---

## Summary of Key Learnings

1. **Transpose all linear layer weights** - Most important gotcha
2. **Use HuggingFace RoPE format** - Don't use Meta format functions
3. **Use ttnn.transformer.scaled_dot_product_attention** - More efficient than manual
4. **Pre-norm with residuals** - Llama architecture pattern
5. **SwiGLU activation** - Not standard ReLU
6. **RMSNorm not LayerNorm** - Different normalization
7. **Grouped-query attention** - Repeat K/V heads
8. **Bisect for debugging** - Faster than layer-by-layer PCC
9. **HuggingFace compatibility** - Elegant generation interface
10. **TILE_LAYOUT + DRAM_MEMORY_CONFIG** - Good defaults for simple models

---

## Appendix: Complete Checklist

When bringing up a new model in ttnn:

- [ ] Load reference model from HuggingFace
- [ ] Understand model architecture (attention type, activation, normalization)
- [ ] Convert embeddings (no transpose needed)
- [ ] Convert linear layer weights (**transpose them!**)
- [ ] Implement normalization layer (RMSNorm/LayerNorm)
- [ ] Implement attention (watch out for RoPE format!)
- [ ] Implement MLP (check activation function)
- [ ] Combine into decoder layer (pre-norm vs post-norm?)
- [ ] Stack layers
- [ ] Add output head
- [ ] Test with simple prompt
- [ ] Debug by bisecting with reference implementation
- [ ] Integrate with HuggingFace generate()
- [ ] Verify output quality
- [ ] Document learnings

---

## Additional Resources

- ttnn documentation: [Insert link to ttnn docs]
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Llama model architecture: https://arxiv.org/abs/2302.13971
- RoPE paper: https://arxiv.org/abs/2104.09864

---

**Good luck with your ttnn model bringup! Remember: take your time, test incrementally, and bisect when debugging.**
