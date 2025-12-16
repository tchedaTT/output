# Llama 3.2 1B ttnn Model Bringup - Learnings & Documentation

This document captures key learnings, challenges, and insights from implementing Llama 3.2 1B in ttnn.

## Overview

The task was to create a simple, clean implementation of Llama 3.2 1B using ttnn that:
- Loads meta-llama/Llama-3.2 1B from HuggingFace
- Converts weights to ttnn format
- Runs both prefill and decode passes
- Is compatible with HuggingFace's `generate()` function
- Uses DRAM interleaved memory (no sharding for simplicity)

## Model Architecture

Llama 3.2 1B has the following configuration:
- **Vocabulary size**: 128,256
- **Hidden size**: 2,048
- **Intermediate size**: 8,192 (for MLP)
- **Number of layers**: 16
- **Number of attention heads**: 32
- **Number of key-value heads**: 8 (Grouped Query Attention)
- **Max position embeddings**: 131,072
- **RMS norm epsilon**: 1e-5
- **RoPE theta**: 500,000.0
- **RoPE scaling**: Llama 3 style with factor=32.0, high_freq_factor=4.0, low_freq_factor=1.0

## Key ttnn Concepts and Functions

### 1. Weight Loading and Conversion

**Critical Discovery**: ttnn.linear expects different weight dimensions than PyTorch!

```python
# PyTorch: (A, B) @ (C, B).T = (A, C)
# ttnn:    (A, B) @ (B, C) = (A, C)
#
# Therefore: ALWAYS TRANSPOSE WEIGHTS before loading!
weight_ttnn = ttnn.from_torch(weight.T, ...)
```

### 2. Memory Configurations

ttnn provides different memory configs:
- `ttnn.DRAM_MEMORY_CONFIG` - Slower but larger capacity
- `ttnn.L1_MEMORY_CONFIG` - Faster but limited size
- Custom sharded configs (for advanced optimization)

For simple bringup, DRAM is sufficient and easier to work with.

### 3. Layout Types

ttnn uses two main layouts:
- **ttnn.TILE_LAYOUT** - 32x32 tiles, optimal for compute operations (matmul, linear, etc.)
- **ttnn.ROW_MAJOR_LAYOUT** - For embeddings and certain weight tensors

**Rule of thumb:**
- Use TILE_LAYOUT for computation tensors
- Use ROW_MAJOR_LAYOUT for embedding weights and RMSNorm weights

### 4. RMSNorm Requirements

**Critical Discovery**: RMSNorm weights have specific shape requirements!

```python
# ttnn.rms_norm expects weights in shape: [1, 1, dim // 32, 32]
TILE_SIZE = 32
weight_reshaped = weight.reshape(1, 1, hidden_size // TILE_SIZE, TILE_SIZE)
weight_ttnn = ttnn.from_torch(
    weight_reshaped,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Usage:
output = ttnn.rms_norm(input, epsilon=eps, weight=weight_ttnn)
```

The error message if you get this wrong:
```
TT_FATAL: Gamma's last padded dim needs to equal tile width and gamma's volume needs to align with last padded dim of input.
```

### 5. Essential ttnn Operations

```python
# Embedding lookup
ttnn.embedding(input_ids, weight_table, layout=ttnn.TILE_LAYOUT)

# Linear layer (remember the transpose!)
ttnn.linear(input, weight_transposed, bias=None, memory_config=...)

# Activation functions
ttnn.silu(x)  # For Llama MLP
ttnn.gelu(x)  # For other models

# Element-wise operations
ttnn.mul(a, b)
ttnn.add(a, b)
ttnn.sub(a, b)

# Tensor manipulation
ttnn.reshape(x, shape)
ttnn.permute(x, dims)
ttnn.transpose(x, dim1, dim2)

# Normalization
ttnn.rms_norm(x, epsilon=eps, weight=w)
ttnn.layer_norm(x, weight=w, bias=b, epsilon=eps)

# Attention (advanced)
ttnn.transformer.scaled_dot_product_attention(...)
```

### 6. Conversions Between ttnn and torch

```python
# torch -> ttnn
x_ttnn = ttnn.from_torch(
    x_torch,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# ttnn -> torch
x_torch = ttnn.to_torch(x_ttnn)
```

**Important**: Avoid conversions in the forward pass for optimal performance. Do weight conversion once during model initialization.

## Implementation Approaches

### Approach 1: Pure ttnn (Ideal but Complex)

Implement every operation in ttnn:
- **Pros**: Best performance, truly native ttnn
- **Cons**: Requires implementing complex operations like RoPE in ttnn, which may not have direct APIs

**Key Challenge**: Rotary Position Embeddings (RoPE)

RoPE requires:
1. Computing cos/sin tables
2. Splitting tensors in half
3. Rotating pairs of elements: `[x1*cos - x2*sin, x1*sin + x2*cos]`

ttnn operations needed:
- `ttnn.slice()` or `ttnn.split()` for splitting tensors
- Element-wise `mul`, `add`, `sub`
- `ttnn.concat()` to recombine

**Gotcha with HuggingFace RoPE format**:
- Task instructions emphasize: Llama 3.2 from HF follows HF conventions, NOT Meta conventions
- tt_transformers code does swizzling/permutation to convert HF -> Meta format
- We should avoid this complexity and work with HF format directly

### Approach 2: Hybrid torch/ttnn (Practical for Bringup)

Use ttnn for major operations, torch for complex transformations:
- Embeddings: ttnn
- Linear layers: ttnn
- RMSNorm: ttnn
- MLP: ttnn
- Attention: hybrid (projections in ttnn, RoPE and attention in torch)
- LM head: ttnn

**Pros**:
- Simpler to implement and debug
- Still demonstrates ttnn usage for main operations
- Can iteratively replace torch operations with ttnn

**Cons**:
- Conversion overhead between torch/ttnn
- Not optimal performance

### Approach 3: HuggingFace Integration (Most Compatible)

Wrap model to be compatible with HuggingFace's `generate()`:
```python
from transformers import PreTrainedModel, GenerationMixin

class TtnnLlamaForCausalLM(PreTrainedModel, GenerationMixin):
    def __init__(self, hf_model, ttnn_device):
        super().__init__(hf_model.config)
        self.config = hf_model.config
        self.generation_config = hf_model.generation_config
        # ... load ttnn weights ...

    def forward(self, input_ids, **kwargs):
        # ... ttnn implementation ...
        return CausalLMOutputWithPast(logits=logits, ...)
```

**Critical gotcha**: `PreTrainedModel` reserves the `device` attribute, so use `ttnn_device` instead!

## Debugging Strategies

### 1. Layer-by-Layer Validation

When encountering correctness issues:

```python
# Create reference wrappers for each component
def reference_mlp(x):
    """Use HF implementation"""
    return hf_layer.mlp(x)

def ttnn_mlp(x):
    """Your ttnn implementation"""
    # ... ttnn code ...

# Compare outputs
torch_out = reference_mlp(input)
ttnn_out = ttnn_to_torch(ttnn_mlp(torch_to_ttnn(input)))
pcc = compute_pcc(torch_out, ttnn_out)  # Pearson Correlation Coefficient
```

This is faster than layer-by-layer PCC for simple models.

### 2. Shape Debugging

Add shape assertions liberally:

```python
print(f"After Q proj: {query_states.shape}")  # Should be [batch, seq, num_heads * head_dim]
print(f"After reshape: {query_states.shape}")  # Should be [batch, seq, num_heads, head_dim]
print(f"After permute: {query_states.shape}")  # Should be [batch, num_heads, seq, head_dim]
```

Common shape issues:
- Forgetting to reshape after linear projections
- Wrong permute dimensions for attention
- Incorrect GQA repeat dimensions

### 3. Device Initialization Issues

If device fails to initialize:
```bash
# Reset the device
tt-smi -r

# Check device status
tt-smi
```

### 4. Memory Issues

If you run out of memory:
- Use DRAM instead of L1
- Reduce batch size
- Process fewer tokens at once
- Check for memory leaks (tensors not being deallocated)

## Common Pitfalls and Solutions

### 1. RMSNorm Shape Mismatch

**Error**: `Gamma's last padded dim needs to equal tile width...`

**Solution**: Reshape weights to `[1, 1, dim // 32, 32]`

### 2. Linear Layer Weight Transpose

**Error**: Dimension mismatch in matmul

**Solution**: Always transpose PyTorch weights before loading:
```python
weight_ttnn = ttnn.from_torch(pytorch_weight.T, ...)
```

### 3. Grouped Query Attention (GQA)

Llama 3.2 1B uses GQA with:
- 32 query heads
- 8 key-value heads
- Ratio: 4 query heads per KV head

**Implementation**:
```python
# After K/V projections and RoPE
if num_kv_heads != num_heads:
    key_states = torch.repeat_interleave(
        key_states,
        num_heads // num_kv_heads,
        dim=1  # heads dimension
    )
    value_states = torch.repeat_interleave(
        value_states,
        num_heads // num_kv_heads,
        dim=1
    )
```

### 4. PreTrainedModel Compatibility

**Error**: `AttributeError: can't set attribute 'device'`

**Cause**: PreTrainedModel reserves certain attribute names

**Solution**: Use different names (e.g., `ttnn_device`, `tt_device`)

### 5. Generation Method Missing

**Error**: `'TtnnLlamaForCausalLM' object has no attribute 'generate'`

**Solution**: Inherit from both `PreTrainedModel` and `GenerationMixin`

## Performance Considerations (Not Implemented in Simple Version)

For production/optimized implementations, consider:

1. **KV Caching**: Store past key/value states to avoid recomputation during decode
2. **Sharding**: Distribute tensors across cores/devices
3. **Flash Attention**: Optimized attention implementation
4. **Mixed Precision**: Use bfp8 where appropriate
5. **Program Caching**: Reuse compiled kernels
6. **Async Operations**: Overlap compute and data movement

## File Structure

```
models/demos/simple/
└── llama32_1b.py          # Main implementation
```

For more complex models, use:
```
models/demos/MODEL_NAME/
├── tt/
│   ├── model.py           # Main model
│   ├── attention.py       # Attention module
│   ├── mlp.py             # MLP module
│   └── embedding.py       # Embeddings
├── tests/
│   ├── test_pcc.py        # Correctness tests
│   └── test_perf.py       # Performance tests
└── demo/
    └── demo.py            # Usage example
```

## Testing Protocol

### 1. Functional Test

```bash
python llama32_1b.py --prompt "1 2 3 4 5 6 7 8 9 10 11 12" --max-new-tokens 15
```

Expected: Should continue the sequence (13 14 15 16...)

### 2. Correctness Validation

Compare outputs with HuggingFace reference:
- Same prompt
- Same generation settings (greedy, no sampling)
- Check if outputs match or are very similar

### 3. Performance Profiling

```python
import ttnn

# Enable profiling
ttnn.enable_program_cache()
ttnn.enable_graph_compilation_cache()

# Run inference
# ...

# Check performance
ttnn.dump_profile()
```

## Key Learnings Summary

### What Would Have Saved Time

1. **RMSNorm shape requirement**: Knowing upfront that weights need `[1, 1, dim//32, 32]` shape would have saved debugging time

2. **Weight transpose requirement**: Understanding that ttnn.linear uses different convention than PyTorch is critical

3. **HuggingFace compatibility**: Knowing about `PreTrainedModel` and `GenerationMixin` from the start would have made integration smoother

4. **RoPE complexity**: RoPE is complex to implement in pure ttnn for a first pass. Hybrid approach is pragmatic for bringup.

5. **Available ttnn operations**: Familiarizing with `ttnn.transformer.*` operations earlier would have helped. There's `scaled_dot_product_attention` available!

### Best Practices for Future Bringups

1. **Start simple**: Get basic forward pass working first, optimize later

2. **Validate incrementally**: Test each component (embedding, layer, head) individually

3. **Use reference implementation**: Keep HuggingFace model loaded for comparison

4. **Print shapes liberally**: Shape mismatches are the most common errors

5. **Read existing implementations**: tt_transformers code is complex but shows advanced patterns

6. **Document gotchas**: Write down non-obvious requirements as you discover them

7. **Hybrid is okay**: For bringup, mixing torch/ttnn is pragmatic. Optimize iteratively.

## Useful Commands

```bash
# Reset device if it gets into bad state
tt-smi -r

# Check device status
tt-smi

# Check ttnn operations available
python -c "import ttnn; print([x for x in dir(ttnn) if not x.startswith('_')])"

# Check transformer operations
python -c "import ttnn; print(dir(ttnn.transformer))"
```

## Resources

- **Tech Reports**: `/tt-metal/tech_reports/` - Contains detailed documentation on:
  - FlashAttention optimization
  - Model bringup workflow (TTNN-model-bringup.md)
  - Performance profiling

- **Reference Implementations**: `/tt-metal/models/tt_transformers/tt/` - Advanced implementations with:
  - Multi-device support
  - Sharding strategies
  - Optimized attention

- **Common Utilities**: `/tt-metal/models/common/` - Reusable components:
  - RMSNorm implementation
  - LightweightModule base class
  - Utility functions

## Next Steps for Full Implementation

To complete a production-ready implementation:

1. **Implement KV caching** for efficient decode
2. **Add pure ttnn RoPE** using available ops
3. **Optimize memory layout** (consider sharding)
4. **Add comprehensive tests** (PCC validation per layer)
5. **Performance profiling** and optimization
6. **Support batched inference**
7. **Add dynamic shape handling**

## Conclusion

Bringing up a model in ttnn requires understanding:
- ttnn's operation conventions (weight formats, shapes)
- Memory and layout management
- Hybrid torch/ttnn approach for rapid prototyping
- Incremental validation strategy

The key is to start simple, validate frequently, and optimize iteratively. The ttnn framework provides powerful operations, but requires attention to details like weight transposes and tensor shapes.

For this bringup, we created a hybrid implementation that:
- Uses ttnn for embeddings, linear layers, RMSNorm, and MLP operations
- Uses torch for RoPE and attention (for simplicity)
- Is compatible with HuggingFace's generate function
- Provides a foundation for future optimization

The implementation demonstrates core ttnn concepts while maintaining simplicity for educational purposes.
