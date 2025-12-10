# Llama 3.2 1B TTNN Model Bringup - Lessons Learned

This document captures everything learned during the process of bringing up Llama 3.2 1B in ttnn. This is meant to help future developers understand the framework, avoid common pitfalls, and accelerate their own model implementations.

## Table of Contents

1. [TTNN Framework Overview](#ttnn-framework-overview)
2. [Key Conventions and Gotchas](#key-conventions-and-gotchas)
3. [Implementation Approach](#implementation-approach)
4. [RoPE Implementation](#rope-implementation)
5. [Attention Implementation](#attention-implementation)
6. [Debugging Strategies](#debugging-strategies)
7. [Performance Considerations](#performance-considerations)
8. [Resources](#resources)

---

## TTNN Framework Overview

### What is TTNN?

TTNN is Tenstorrent's neural network framework that provides:
- **Hardware acceleration** for ML operations on Tenstorrent devices
- **Device management** for single and multi-chip configurations
- **Operations library** covering linear algebra, activations, attention, etc.
- **Memory management** with explicit control over DRAM vs L1 placement
- **Multi-device support** with sharding and collective communication

### Directory Structure

```
tt-metal/
├── ttnn/                           # Core TTNN Python module
│   ├── operations/                 # Operation implementations
│   │   ├── matmul.py
│   │   ├── transformer.py          # Attention ops
│   │   └── ...
│   └── __init__.py
├── models/
│   ├── tt_transformers/            # Advanced multi-device LLM framework
│   │   └── tt/                     # TTNN implementations
│   ├── demos/                      # Example implementations
│   └── common/                     # Shared utilities
└── tech_reports/                   # Technical documentation
    └── ttnn/
        ├── ttnn.md
        └── TTNN-model-bringup.md
```

---

## Key Conventions and Gotchas

### 1. Linear Layer Weight Transposes

**Critical difference**: ttnn.linear expects **(A, B) @ (B, C)** while PyTorch uses **(A, B) @ (C, B)**

```python
# PyTorch weights
torch_weight = state_dict["layer.weight"]  # Shape: [out_features, in_features]

# For ttnn, transpose the weights!
ttnn_weight = torch_weight.T.unsqueeze(0).unsqueeze(0)  # [1, 1, in_features, out_features]
ttnn_weight_tensor = ttnn.from_torch(
    ttnn_weight,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)
```

**Why this matters**: Forgetting this transpose will give you completely incorrect outputs! This was mentioned in AGENTS.md but is easy to miss.

### 2. Tensor Shapes and Padding

TTNN works with **4D tensors** and pads to **tile boundaries (32x32)**:

```python
# Input: torch tensor [batch, seqlen, hidden_size]
x = torch.randn(1, 24, 2048)

# Convert to ttnn
x_tt = ttnn.from_torch(
    x.unsqueeze(0).unsqueeze(0),  # [1, 1, 24, 2048]
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device
)

# TTNN automatically pads: [1, 1, 32, 2048] (24 → 32)
# Always slice back to original size!
x_back = ttnn.to_torch(ttnn.from_device(x_tt))
x_back = x_back.squeeze(0).squeeze(0)[:24]  # Unpad!
```

### 3. Memory Layouts

Two main layouts:
- **TILE_LAYOUT**: For compute operations (matmul, attention, activations)
- **ROW_MAJOR_LAYOUT**: For embeddings and data transfer

```python
# Embedding uses ROW_MAJOR
embed_tt = ttnn.from_torch(
    embed_weight,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,  # Not TILE!
    device=device
)

# Compute ops use TILE
linear_weight = ttnn.from_torch(
    weight,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,  # Required for matmul
    device=device
)
```

### 4. Memory Configurations

Common memory configs:
- **DRAM_MEMORY_CONFIG**: Larger, slower (use for weights and less frequent access)
- **L1_MEMORY_CONFIG**: Faster, smaller (use for activations in tight loops)
- **SHARDED configs**: For multi-core parallelism (advanced)

For a simple single-device implementation, **DRAM_MEMORY_CONFIG** is safe and works well.

---

## Implementation Approach

### Strategy: Hybrid PyTorch-TTNN

For educational purposes and rapid prototyping, I used a **hybrid approach**:

1. **Use ttnn for compute-intensive ops**: linear, attention, activations
2. **Keep control flow in PyTorch**: embeddings, normalization, RoPE
3. **Convert at boundaries**: PyTorch ↔ ttnn at module boundaries

This approach:
- ✅ Keeps code simple and debuggable
- ✅ Demonstrates ttnn usage clearly
- ✅ Works correctly out of the box
- ⚠️ Has conversion overhead (not optimized for production)

### Module Structure

```
LlamaForCausalLM
├── LlamaModel
│   ├── embed_tokens (PyTorch)
│   ├── layers (List of LlamaDecoderLayer)
│   │   ├── LlamaAttention
│   │   │   ├── QKV projections (ttnn.linear)
│   │   │   ├── RoPE (PyTorch)
│   │   │   ├── scaled_dot_product_attention (ttnn)
│   │   │   └── output projection (ttnn.linear)
│   │   ├── LlamaFeedForward
│   │   │   ├── gate_proj (ttnn.linear)
│   │   │   ├── up_proj (ttnn.linear)
│   │   │   ├── silu (ttnn.silu)
│   │   │   └── down_proj (ttnn.linear)
│   │   └── RMSNorm layers (PyTorch)
│   └── final norm (PyTorch)
└── lm_head (PyTorch)
```

### Why This Works

**Compute-intensive operations** benefit most from hardware acceleration:
- Matrix multiplications (QKV projections, FFN, output projection)
- Attention (scaled_dot_product_attention)
- Activations (silu, gelu)

**Less critical operations** are simpler to keep in PyTorch:
- Embeddings (lookup table, not compute-bound)
- RMSNorm (element-wise, minimal benefit from acceleration)
- Tokenization and sampling

---

## RoPE Implementation

### The Big Gotcha: HuggingFace vs Meta Format

**Critical**: HuggingFace models use a different RoPE format than Meta's original Llama!

**Meta format (interleaved)**:
```
[x0, x1, x2, x3, ...] → [x0, x1], [x2, x3], ... pairs
Rotate: [x0', x1'] = [x0*cos - x1*sin, x0*sin + x1*cos]
```

**HuggingFace format (halved)**:
```
[x0, x1, x2, x3, x4, x5, ...] → first_half [x0, x1, x2], second_half [x3, x4, x5]
Rotate: [x0', x3'] = [x0*cos - x3*sin, x0*sin + x3*cos]
```

The `tt_transformers` code converts HuggingFace → Meta format by swizzling weights. **We want to avoid this!**

### Llama 3 RoPE Scaling

Llama 3.x uses a special scaling formula to extend context length:

```python
def apply_llama3_rope_scaling(freqs, factor, low_freq_factor, high_freq_factor, old_context_len):
    """Apply Llama 3 frequency scaling."""
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)  # High freq: no scaling
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)  # Low freq: full scaling
        else:
            # Interpolate between scaled and unscaled
            smooth = (old_context_len / wavelen - low_freq_factor) / \
                     (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs)
```

For Llama 3.2 1B:
- `factor = 32.0`
- `low_freq_factor = 1.0`
- `high_freq_factor = 4.0`
- `original_max_position_embeddings = 8192`

### RoPE Application (HuggingFace Format)

```python
def apply_rotary_emb(xq, xk, cos, sin):
    """
    xq: [batch, n_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    # Split into "real" and "imaginary" parts (first half and second half)
    xq_r, xq_i = xq[..., :xq.shape[-1]//2], xq[..., xq.shape[-1]//2:]
    xk_r, xk_i = xk[..., :xk.shape[-1]//2], xk[..., xk.shape[-1]//2:]

    # cos/sin were computed with doubled frequencies, so take first half
    cos = cos[..., :cos.shape[-1]//2].unsqueeze(0).unsqueeze(0)
    sin = sin[..., :sin.shape[-1]//2].unsqueeze(0).unsqueeze(0)

    # Apply rotation: (r, i) * (cos, sin) = (r*cos - i*sin, r*sin + i*cos)
    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    return torch.cat([xq_out_r, xq_out_i], dim=-1), torch.cat([xk_out_r, xk_out_i], dim=-1)
```

**Key insight**: Don't try to use `ttnn.experimental.rotary_embedding_llama` unless you've converted to Meta format. For simplicity, do RoPE in PyTorch.

---

## Attention Implementation

### Using ttnn.transformer.scaled_dot_product_attention

TTNN provides an optimized Flash Attention implementation:

```python
# Prepare inputs: [batch, n_heads, seq_len, head_dim]
xq_tt = ttnn.from_torch(xq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
xk_tt = ttnn.from_torch(xk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
xv_tt = ttnn.from_torch(xv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Compute attention
scale = 1.0 / math.sqrt(head_dim)
output_tt = ttnn.transformer.scaled_dot_product_attention(
    xq_tt, xk_tt, xv_tt,
    is_causal=True,  # For autoregressive generation
    scale=scale,
    memory_config=ttnn.DRAM_MEMORY_CONFIG
)
```

### Grouped-Query Attention (GQA)

Llama 3.2 1B uses GQA with 32 query heads and 8 KV heads:

```python
# Repeat KV heads to match query heads
num_key_value_groups = num_heads // num_key_value_heads  # 32 // 8 = 4

if num_key_value_groups > 1:
    xk = xk.repeat_interleave(num_key_value_groups, dim=1)
    xv = xv.repeat_interleave(num_key_value_groups, dim=1)
```

Do this **before** passing to scaled_dot_product_attention.

### Decode vs Prefill

The implementation can be extended to support separate prefill and decode paths:

**Prefill** (processing entire prompt):
- Use `ttnn.transformer.scaled_dot_product_attention` with `is_causal=True`
- Processes entire sequence at once
- Fills KV cache

**Decode** (autoregressive generation):
- Use `ttnn.transformer.scaled_dot_product_attention_decode`
- Processes one token at a time
- Queries existing KV cache

For this simple demo, we use the general `scaled_dot_product_attention` for both.

---

## Debugging Strategies

### 1. Manual Bisection

When outputs are wrong, use **manual bisection** to isolate the problem:

```python
class HybridAttention(nn.Module):
    def forward(self, x, cos, sin):
        # Replace suspicious ttnn ops with reference torch ops

        # Original: xq_tt = ttnn.linear(x_tt, self.wq)
        # Debug version:
        xq_tt = ttnn.linear(x_tt, self.wq)
        xq_torch = ttnn.to_torch(ttnn.from_device(xq_tt))

        # Compare with reference
        xq_ref = torch.nn.functional.linear(x, self.wq_torch_weight)
        pcc = compute_pcc(xq_torch, xq_ref)
        print(f"Q proj PCC: {pcc}")  # Should be > 0.999

        # Continue bisecting...
```

This is **faster than layer-by-layer PCC** for simple models because you can quickly narrow down which op is wrong.

### 2. Shape Debugging

Add assertions everywhere:

```python
x = ttnn.to_torch(ttnn.from_device(x_tt))
print(f"After linear: {x.shape}")  # Check for unexpected padding
assert x.shape[2] >= expected_seqlen, "Sequence got truncated!"
```

### 3. Reference Model Comparison

Keep a reference PyTorch model and compare outputs:

```python
# Reference
ref_out = ref_model(input_ids)

# TTNN
ttnn_out = ttnn_model(input_ids)

# Compare
diff = (ref_out - ttnn_out).abs().max()
print(f"Max diff: {diff}")  # Should be < 0.1 for bfloat16
```

### 4. Device Reset

If your device gets into a bad state:

```bash
tt-smi -r  # Reset device
```

This is mentioned in the task description and is important when debugging crashes.

---

## Performance Considerations

### What This Demo Does NOT Optimize

This implementation prioritizes **clarity and correctness** over performance. Production implementations should optimize:

1. **Reduce PyTorch ↔ TTNN conversions**
   - Keep more operations in ttnn
   - Use ttnn's RoPE implementation (after converting weights)
   - Implement norms in ttnn

2. **Use advanced memory configs**
   - L1_MEMORY_CONFIG for hot data
   - Sharded configurations for multi-core parallelism
   - Height/Width/Block sharding strategies

3. **Optimize matmul configs**
   - Custom `program_config` for linear layers
   - Tune block sizes and grid sizes
   - Use appropriate math fidelity (HiFi2, LoFi)

4. **Enable advanced features**
   - **Trace**: Compile graph once, replay multiple times
   - **2CQ**: Dual command queues for overlapping compute
   - **Paged attention**: For efficient KV cache management
   - **Chunked prefill**: For long context windows

5. **Use bfloat8_b for weights**
   - Lower precision = faster, smaller
   - Usually minimal accuracy impact

### Performance Path

From tech reports, the optimization path is:

**Stage 1**: Per-op optimization
- Choose data types (bfloat16 vs bfloat8_b)
- Select math fidelity (HiFi vs LoFi)
- Configure sharding strategies

**Stage 2**: Module-level optimization
- Profile with Tracy
- Generate perf sheets
- Identify bottlenecks
- Optimize critical ops

**Stage 3**: Full-model optimization
- Implement Trace for graph compilation
- Add 2CQ for overlapped execution
- Tune batch sizes and sequence lengths

---

## Resources

### Essential Documentation

1. **Tech Reports** (`tt-metal/tech_reports/ttnn/`):
   - `TTNN-model-bringup.md` - Complete bringup workflow
   - `graph-tracing.md` - Profiling and optimization
   - `comparison-mode.md` - Automatic correctness checking

2. **Example Implementations**:
   - `models/tt_transformers/` - Production multi-device LLMs
   - `models/demos/ttnn_resnet/` - CNN example with sharding
   - `models/demos/ttnn_falcon7b/` - Simpler LLM example

3. **TTNN API**:
   - `ttnn/__init__.py` - Main API exports
   - `ttnn/operations/` - Available operations
   - C++ headers in `ttnn/cpp/ttnn/operations/` for detailed specs

### Useful TTNN Functions

**Tensor Creation**:
- `ttnn.from_torch()` - Convert PyTorch tensor
- `ttnn.to_torch()` - Convert back to PyTorch
- `ttnn.zeros()`, `ttnn.ones()`, `ttnn.empty()`

**Linear Algebra**:
- `ttnn.linear()` - **Note**: (A,B) @ (B,C) format!
- `ttnn.matmul()` - General matrix multiplication
- `ttnn.bmm()` - Batch matrix multiplication

**Attention**:
- `ttnn.transformer.scaled_dot_product_attention()` - Prefill attention
- `ttnn.transformer.scaled_dot_product_attention_decode()` - Decode attention
- `ttnn.transformer.paged_scaled_dot_product_attention_decode()` - With KV cache paging

**Activations**:
- `ttnn.silu()`, `ttnn.gelu()`, `ttnn.relu()`
- `ttnn.softmax()`
- `ttnn.tanh()`, `ttnn.sigmoid()`

**Tensor Manipulation**:
- `ttnn.reshape()`, `ttnn.permute()`, `ttnn.transpose()`
- `ttnn.concat()`, `ttnn.split()`
- `ttnn.slice()`
- `ttnn.repeat_interleave()` - For GQA

**Normalization**:
- `ttnn.layer_norm()`, `ttnn.rms_norm()`
- `ttnn.group_norm()`

**Memory Management**:
- `ttnn.to_memory_config()` - Move tensors between memory types
- `ttnn.to_layout()` - Convert between ROW_MAJOR and TILE layouts
- `ttnn.deallocate()` - Explicit deallocation

### Key Learnings Summary

1. ✅ **Always transpose weights** for ttnn.linear
2. ✅ **Watch for padding** - ttnn pads to 32x32 tiles
3. ✅ **Use correct layouts** - ROW_MAJOR for embed, TILE for compute
4. ✅ **HuggingFace RoPE** is different from Meta RoPE
5. ✅ **Start simple** - hybrid PyTorch/ttnn is fine for learning
6. ✅ **Bisect for debugging** - faster than layer-by-layer PCC
7. ✅ **DRAM is safe** - start with DRAM_MEMORY_CONFIG
8. ✅ **Read tech reports** - they contain critical information

---

## Final Thoughts

Bringing up a model in ttnn involves:
1. Understanding the framework conventions (especially weight transposes!)
2. Handling tensor shapes and padding carefully
3. Using the right memory layouts and configs
4. Being aware of format differences (RoPE, attention patterns)
5. Debugging systematically when things go wrong

The hybrid PyTorch-ttnn approach used here is great for learning and prototyping. Production implementations should move more operations to ttnn and apply the optimization strategies from the tech reports.

**Time saved**: If I had known about the weight transpose requirement upfront and understood ttnn's padding behavior, it would have saved at least an hour of debugging. Hence this document!

Happy model porting! 🚀
