# Llama 3.2 1B Model Bringup Guide

This document captures everything learned during the process of implementing Llama 3.2 1B in ttnn, including ttnn conventions, gotchas, debugging approaches, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [Key ttnn Concepts](#key-ttnn-concepts)
3. [Implementation Approach](#implementation-approach)
4. [Important Gotchas](#important-gotchas)
5. [Debugging Strategies](#debugging-strategies)
6. [Performance Considerations](#performance-considerations)
7. [Future Improvements](#future-improvements)

---

## Overview

The goal was to create a simple, clean implementation of Llama 3.2 1B that:
- Loads the model from HuggingFace (`meta-llama/Llama-3.2-1B`)
- Converts weights to ttnn format
- Supports both prefill and decode passes
- Works with HuggingFace's `generate()` function
- Uses ttnn operations directly (not copying large code sections from tt_transformers)

**Model Architecture:**
- **Layers:** 16 transformer blocks
- **Hidden size:** 2048
- **Attention heads:** 32 (with 8 KV heads for GQA)
- **Intermediate size:** 8192
- **Vocab size:** 128256
- **RoPE theta:** 500000.0

---

## Key ttnn Concepts

### 1. Tensor Conversion

**PyTorch → ttnn:**
```python
ttnn_tensor = ttnn.from_torch(
    torch_tensor,
    dtype=ttnn.bfloat16,          # or bfloat8_b, bfloat4_b, float32, uint32, int32
    layout=ttnn.TILE_LAYOUT,      # or ROW_MAJOR_LAYOUT
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG  # or L1_MEMORY_CONFIG
)
```

**ttnn → PyTorch:**
```python
torch_tensor = ttnn.to_torch(ttnn_tensor)
```

### 2. Weight Matrix Transposition

**CRITICAL:** `ttnn.linear` expects weights in **OPPOSITE** format from PyTorch!

- **PyTorch:** `torch.nn.functional.linear(input, weight)` expects:
  - Input: `[..., in_features]`
  - Weight: `[out_features, in_features]`

- **ttnn:** `ttnn.linear(input, weight)` expects:
  - Input: `[..., in_features]`
  - Weight: `[in_features, out_features]`

**Example:**
```python
# Load PyTorch weight: shape [out_features, in_features]
torch_weight = state_dict["linear.weight"]

# MUST transpose for ttnn!
ttnn_weight = ttnn.from_torch(
    torch_weight.T,  # <- Critical transpose!
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# Now can use with ttnn.linear
output = ttnn.linear(input, ttnn_weight)
```

### 3. Memory Layouts

**TILE_LAYOUT (32x32 tiles):**
- Required for most compute operations (matmul, attention, etc.)
- Data is organized in 32x32 tiles for hardware efficiency
- Best for matrix operations

**ROW_MAJOR_LAYOUT:**
- Standard row-major format
- Used for embeddings, norms
- Better for element-wise operations

### 4. Memory Configurations

- **`ttnn.DRAM_MEMORY_CONFIG`:** Off-chip DRAM (slower but larger capacity)
- **`ttnn.L1_MEMORY_CONFIG`:** On-chip L1 (faster but smaller capacity)
- **Sharded configs:** For advanced multi-core parallelism (not needed for simple implementation)

### 5. Data Types

- **`ttnn.bfloat16`:** 16-bit brain float (most common for weights and activations)
- **`ttnn.bfloat8_b`:** 8-bit brain float (for activation compression)
- **`ttnn.bfloat4_b`:** 4-bit brain float (for weight compression)
- **`ttnn.float32`:** 32-bit float
- **`ttnn.uint32`, `ttnn.int32`:** Integer types (for token IDs, indices)

---

## Implementation Approach

### 1. Model Structure

The implementation follows standard Llama architecture:

```
LlamaForCausalLM
├── Embedding (token → hidden states)
├── N x LlamaDecoderLayer
│   ├── Input LayerNorm (RMSNorm)
│   ├── LlamaAttention (QKV proj, RoPE, SDPA, output proj)
│   ├── Post-Attention LayerNorm (RMSNorm)
│   └── LlamaMLP (gate, up, down projections with SiLU)
├── Final LayerNorm (RMSNorm)
└── LM Head (project to vocab)
```

### 2. Attention Implementation

**Key steps:**
1. **QKV Projection:** Concatenate Q, K, V weights into single matrix for efficiency
2. **Split QKV:** Separate concatenated output into Q, K, V tensors
3. **Reshape for Multi-Head:** Split hidden dimension into (num_heads, head_dim)
4. **Apply RoPE:** Rotary position embeddings (see RoPE section below)
5. **Handle GQA:** Repeat K, V to match Q heads (Grouped Query Attention)
6. **SDPA:** Use `ttnn.transformer.scaled_dot_product_attention`
7. **Concat Heads:** Merge heads back to hidden dimension
8. **Output Projection:** Final linear layer

**Code pattern:**
```python
# QKV projection (concatenated weights)
xqkv = ttnn.linear(x, self.wqkv)

# Split and reshape (currently using torch for simplicity)
xqkv_torch = ttnn.to_torch(xqkv)
q, k, v = split_qkv(xqkv_torch)  # Custom splitting logic

# Apply RoPE
q, k = apply_rotary_emb(q, k, cos, sin)

# GQA: repeat K, V
k = k.repeat_interleave(num_key_value_groups, dim=1)
v = v.repeat_interleave(num_key_value_groups, dim=1)

# Convert back to ttnn
q_ttnn = ttnn.from_torch(q, ...)
k_ttnn = ttnn.from_torch(k, ...)
v_ttnn = ttnn.from_torch(v, ...)

# Attention
scale = 1.0 / math.sqrt(head_dim)
attn_output = ttnn.transformer.scaled_dot_product_attention(
    q_ttnn, k_ttnn, v_ttnn,
    is_causal=True,
    scale=scale
)

# Reshape and output projection
attn_reshaped = reshape_and_concat_heads(attn_output)
output = ttnn.linear(attn_reshaped, self.wo)
```

### 3. MLP Implementation

**Formula:** `down_proj(silu(gate_proj(x)) * up_proj(x))`

```python
def forward(self, x):
    # Gate projection + SiLU activation
    gate = ttnn.linear(x, self.w1)
    gate = ttnn.silu(gate)

    # Up projection
    up = ttnn.linear(x, self.w3)

    # Element-wise multiply
    intermediate = ttnn.mul(gate, up)

    # Down projection
    output = ttnn.linear(intermediate, self.w2)

    return output
```

### 4. RMSNorm

ttnn provides `ttnn.rms_norm` out of the box:

```python
normalized = ttnn.rms_norm(
    input_tensor,
    weight_tensor,
    eps=1e-5  # or config.rms_norm_eps
)
```

**Note:** Weight should use `ROW_MAJOR_LAYOUT`, not `TILE_LAYOUT`.

### 5. Embedding Lookup

Use `ttnn.embedding` for token embedding:

```python
# Input IDs must be uint32
input_ids_ttnn = ttnn.from_torch(
    input_ids,
    dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# Embedding lookup
hidden_states = ttnn.embedding(
    input_ids_ttnn,
    embedding_weight,
    layout=ttnn.TILE_LAYOUT  # Output layout
)
```

---

## Important Gotchas

### 1. RoPE Format: HuggingFace vs Meta

**⚠️ CRITICAL GOTCHA:** The `tt_transformers` code converts HuggingFace RoPE to Meta format by "swizzling" the frequencies. This is done in the `permute_to_meta_format` function:

```python
# tt_transformers/tt/rope.py - permute_to_meta_format
def permute_to_meta_format(cos, sin):
    # Undo the HF permute
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    # ... (similar for sin)
```

**For HuggingFace models like Llama 3.2, we should NOT do this swizzling!** The model already uses HuggingFace RoPE format.

**HuggingFace RoPE format:**
```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
freqs = torch.outer(positions, inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)  # Concatenate with itself
cos = emb.cos()
sin = emb.sin()

# Apply: x_rotated = x * cos + rotate_half(x) * sin
def rotate_half(x):
    x1, x2 = x[..., :dim//2], x[..., dim//2:]
    return torch.cat((-x2, x1), dim=-1)
```

**Lesson:** When working with HuggingFace models, avoid using the `llama` version of rotary_embedding ops from tt_transformers that expect Meta format.

### 2. Tensor Shape Requirements

ttnn operations often require specific shape constraints:

- **Tile alignment:** Dimensions should ideally be multiples of 32 for `TILE_LAYOUT`
- **Batch size:** Some operations work best with specific batch sizes
- **Sequence length:** May need padding for certain sequence lengths

### 3. Memory Management

- **Deallocation:** Use `ttnn.deallocate(tensor)` to free memory when done with tensors
- **Conversions:** Minimize conversions between ttnn ↔ torch (they're expensive)
- **Memory config:** Choose DRAM vs L1 based on tensor size and access patterns

### 4. HuggingFace Integration

To make the model work with HuggingFace's `generate()`:

**Required methods:**
- `forward(input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False)`
- `prepare_inputs_for_generation(input_ids, past_key_values=None, **kwargs)`
- `_reorder_cache(past_key_values, beam_idx)`

**Return format:**
```python
from transformers.modeling_outputs import CausalLMOutputWithPast

return CausalLMOutputWithPast(
    logits=logits_torch,  # Must be torch tensor
    past_key_values=past_key_values,
)
```

---

## Debugging Strategies

### 1. Hybrid Torch/ttnn Debugging

**Strategy:** When output is wrong, systematically replace ttnn operations with torch equivalents to isolate the issue.

**Example:**
```python
# Original (not working correctly)
def attention(x_ttnn):
    q, k, v = qkv_projection_ttnn(x_ttnn)
    attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, ...)
    return output_projection_ttnn(attn)

# Debug: Replace one component at a time
def attention(x_ttnn):
    # Use reference torch implementation for QKV
    x_torch = ttnn.to_torch(x_ttnn)
    q_torch, k_torch, v_torch = reference_qkv_projection(x_torch)
    q_ttnn = ttnn.from_torch(q_torch, ...)

    # Keep ttnn SDPA
    attn = ttnn.transformer.scaled_dot_product_attention(q_ttnn, k_ttnn, v_ttnn, ...)

    # Use reference output projection
    attn_torch = ttnn.to_torch(attn)
    output_torch = reference_output_projection(attn_torch)
    return ttnn.from_torch(output_torch, ...)
```

This binary search approach quickly identifies which component is causing issues.

### 2. Compare Against Reference Model

**Run reference HuggingFace model alongside:**
```python
# Reference forward pass
ref_outputs = hf_model(input_ids)
ref_logits = ref_outputs.logits

# Your ttnn implementation
ttnn_outputs = ttnn_model(input_ids)
ttnn_logits = ttnn_outputs.logits

# Compare
diff = (ref_logits - ttnn_logits).abs()
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")

# Check if token predictions match
ref_tokens = ref_logits.argmax(dim=-1)
ttnn_tokens = ttnn_logits.argmax(dim=-1)
print(f"Token match: {(ref_tokens == ttnn_tokens).all()}")
```

### 3. Layer-by-Layer Validation

**Insert checks after each layer:**
```python
ref_hidden = ref_model.model.embed_tokens(input_ids)
ttnn_hidden = ttnn_model.embed_tokens(input_ids_ttnn)

for i, (ref_layer, ttnn_layer) in enumerate(zip(ref_model.model.layers, ttnn_model.layers)):
    ref_hidden = ref_layer(ref_hidden)[0]
    ttnn_hidden = ttnn_layer(ttnn_hidden, position_ids)

    # Compare
    ttnn_hidden_torch = ttnn.to_torch(ttnn_hidden)
    diff = (ref_hidden - ttnn_hidden_torch).abs()
    print(f"Layer {i} max diff: {diff.max():.6f}")

    # Fail fast if divergence is too large
    assert diff.max() < threshold, f"Layer {i} diverged!"
```

### 4. Device Reset

If the device gets into a bad state (common during development):

```bash
# Reset Wormhole device
tt-smi -r
```

Symptoms of bad device state:
- Initialization failures
- Hanging operations
- Unexpected errors about device state

---

## Performance Considerations

### Current Implementation (Simple)

The current implementation prioritizes **simplicity and correctness** over performance:

✅ **Pros:**
- Easy to understand and debug
- Clean code structure
- Works with HuggingFace ecosystem

❌ **Areas for optimization:**
- Uses DRAM interleaved (not sharded)
- Frequent ttnn ↔ torch conversions
- No KV cache optimization for decode
- Single device only

### What's Not Implemented (Intentionally)

As per task requirements, the following optimizations were skipped:

- **Multi-device support:** Keeps code simple
- **Sharding:** DRAM interleaved is sufficient for 1B model
- **Advanced memory configs:** Would complicate the example
- **Optimized program configs:** Uses defaults
- **KV cache management:** Would require decode mode optimization

---

## Future Improvements

### 1. Optimize RoPE with Pure ttnn

Currently using torch for RoPE. Could optimize with:

```python
# Use ttnn.embedding to gather cos/sin based on position IDs
position_ids_ttnn = ttnn.from_torch(position_ids, dtype=ttnn.uint32, ...)
cos = ttnn.embedding(position_ids_ttnn, self.cos_matrix, ...)
sin = ttnn.embedding(position_ids_ttnn, self.sin_matrix, ...)

# Use ttnn.experimental.rotary_embedding_llama
# But be careful about HF vs Meta format!
q_rotated = ttnn.experimental.rotary_embedding_llama(
    q_heads, cos, sin, transformation_mat, is_decode_mode=False
)
```

### 2. Reduce ttnn ↔ torch Conversions

Current attention implementation converts to torch for splitting/reshaping. Could use:

- `ttnn.experimental.nlp_create_qkv_heads` - Split QKV and reshape to heads
- `ttnn.experimental.nlp_concat_heads` - Merge heads back
- `ttnn.reshape`, `ttnn.transpose`, `ttnn.slice` - Pure ttnn tensor operations

### 3. Add KV Cache for Decode

For efficient autoregressive generation:

```python
# Initialize cache
self.cache_k = ttnn.allocate_cache(...)
self.cache_v = ttnn.allocate_cache(...)

# Update cache
ttnn.experimental.paged_update_cache(
    self.cache_k, new_k, update_idxs_tensor=current_pos
)

# Use cached values in attention
attn = ttnn.transformer.scaled_dot_product_attention_decode(
    q, self.cache_k, self.cache_v,
    cur_pos_tensor=current_pos,
    scale=scale
)
```

### 4. Add Prefill/Decode Mode Splitting

Optimize differently for prefill (seq_len > 1) vs decode (seq_len = 1):

```python
if is_prefill:
    # Use ttnn.transformer.scaled_dot_product_attention
    # Process entire prompt at once
else:
    # Use ttnn.transformer.scaled_dot_product_attention_decode
    # Process one token, use KV cache
```

### 5. Sharding for Larger Models

For models > 1B, consider:

- **Tensor parallel:** Shard weight matrices across devices
- **Pipeline parallel:** Distribute layers across devices
- **Sharded memory configs:** Distribute data across cores

---

## Useful ttnn Operations Reference

### Matrix Operations
```python
ttnn.linear(input, weight, bias=None)  # Linear layer
ttnn.matmul(a, b)                      # Matrix multiply
ttnn.addmm(input, mat1, mat2, alpha=1.0, beta=1.0)  # Fused add + matmul
```

### Element-wise Operations
```python
ttnn.add(a, b)        # Addition (also: a + b)
ttnn.mul(a, b)        # Multiplication (also: a * b)
ttnn.subtract(a, b)   # Subtraction
ttnn.divide(a, b)     # Division
```

### Activations
```python
ttnn.silu(x)          # SiLU/Swish
ttnn.gelu(x)          # GELU
ttnn.relu(x)          # ReLU
ttnn.softmax(x, dim)  # Softmax
```

### Normalization
```python
ttnn.rms_norm(x, weight, eps=1e-5)        # RMSNorm
ttnn.layer_norm(x, weight, bias, eps=1e-5)  # LayerNorm
```

### Attention
```python
# Prefill attention
ttnn.transformer.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    is_causal=True,
    scale=None,
    sliding_window_size=None
)

# Decode attention (with KV cache)
ttnn.transformer.scaled_dot_product_attention_decode(
    q, k, v,
    cur_pos_tensor,
    scale=None,
    sliding_window_size=None
)
```

### Tensor Manipulation
```python
ttnn.reshape(x, shape)              # Reshape
ttnn.transpose(x, dim0, dim1)       # Transpose
ttnn.permute(x, dims)               # Permute dimensions
ttnn.concat(tensors, dim)           # Concatenate
ttnn.slice(x, starts, ends)         # Slice tensor
ttnn.unsqueeze_to_4D(x)            # Add dimensions to make 4D
```

### Embedding
```python
ttnn.embedding(input_ids, weight, layout=ttnn.TILE_LAYOUT)
```

### Device Operations
```python
device = ttnn.open_device(device_id=0)
ttnn.close_device(device)
ttnn.to_device(tensor, device, memory_config)
ttnn.from_device(tensor)
```

---

## Common Error Messages and Solutions

### 1. "Shape not tile-aligned"

**Problem:** Tensor dimensions not multiples of 32 for TILE_LAYOUT.

**Solution:** Pad tensor or use ROW_MAJOR_LAYOUT.

### 2. "Device not initialized"

**Problem:** Device not opened or in bad state.

**Solution:**
```bash
tt-smi -r  # Reset device
```

### 3. "Memory allocation failed"

**Problem:** Out of L1 or DRAM memory.

**Solution:**
- Use DRAM_MEMORY_CONFIG instead of L1
- Deallocate unused tensors
- Reduce batch size

### 4. "Incompatible memory configs"

**Problem:** Operation requires inputs with specific memory configs.

**Solution:**
```python
x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
```

---

## Summary of Key Learnings

1. **Weight Transposition:** Always transpose PyTorch weights for ttnn.linear
2. **RoPE Format:** Be careful with HuggingFace vs Meta format - don't swizzle unnecessarily
3. **Hybrid Debugging:** Mix torch and ttnn operations to isolate issues quickly
4. **Memory Management:** Choose appropriate memory configs, deallocate when possible
5. **Simplicity First:** Get it working correctly before optimizing
6. **Device Resets:** Use `tt-smi -r` when device acts up
7. **HuggingFace Compat:** Implement required methods to use `generate()`
8. **Layout Matters:** TILE_LAYOUT for compute, ROW_MAJOR for element-wise/embedding
9. **Documentation:** ttnn is well-documented in tech_reports and source code
10. **Community Patterns:** Learn from tt_transformers but adapt for your use case

---

## Resources

- **ttnn Documentation:** `tt-metal/ttnn/README.md`
- **Tech Reports:** `tt-metal/tech_reports/`
- **tt_transformers Examples:** `tt-metal/models/tt_transformers/`
- **SDPA Reference:** `tt-metal/ttnn/cpp/ttnn/operations/transformer/`
- **Example Models:** `tt-metal/models/demos/`

---

**Note:** This guide reflects the learning process for bringing up Llama 3.2 1B. Your mileage may vary with different models or architectures, but the core concepts and debugging strategies should transfer well.

Good luck with your ttnn model implementations!
