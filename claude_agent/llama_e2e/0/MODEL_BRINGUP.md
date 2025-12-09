# Llama 3.2 1B Model Bringup in ttnn

This document captures the key learnings, approaches, and gotchas from implementing Llama 3.2 1B in ttnn.

## Overview

The implementation in `llama32_1b.py` provides a clean, educational example of how to port a HuggingFace transformer model to ttnn while maintaining compatibility with HuggingFace's `generate()` function.

## Key ttnn Concepts and Functions

### 1. **ttnn.linear Weight Format**

**CRITICAL DIFFERENCE**: This is the most important gotcha when porting from PyTorch!

- **PyTorch**: `torch.nn.functional.linear` expects weights shaped as `(out_features, in_features)`
- **ttnn**: `ttnn.linear` expects weights shaped as `(in_features, out_features)`

**Solution**: Always transpose PyTorch weights before passing to ttnn:
```python
# PyTorch weight: (out_features, in_features)
weight_tt = ttnn.from_torch(pytorch_linear.weight.T, device=device)  # Note the .T
output_tt = ttnn.linear(input_tt, weight_tt)
```

### 2. **Tensor Conversion Between PyTorch and ttnn**

Two essential functions for bridging PyTorch and ttnn:

```python
# PyTorch -> ttnn
tensor_tt = ttnn.from_torch(tensor_torch, device=device)

# ttnn -> PyTorch
tensor_torch = ttnn.to_torch(tensor_tt)
```

**When to convert**:
- Convert to ttnn before operations you want accelerated
- Convert back to PyTorch for operations not yet available or convenient in ttnn
- Operations like reshaping, slicing, and certain tensor manipulations are often easier in PyTorch

### 3. **ttnn Activation Functions**

Common activations available in ttnn:
- `ttnn.silu(x)` - SiLU/Swish activation (used in Llama's SwiGLU)
- `ttnn.gelu(x)` - GELU activation
- `ttnn.relu(x)` - ReLU activation

### 4. **ttnn.transformer.scaled_dot_product_attention**

Instead of implementing attention manually, use the built-in function:

```python
attn_output = ttnn.transformer.scaled_dot_product_attention(
    query,              # [batch, num_heads, seq_len, head_dim]
    key,                # [batch, num_heads, kv_seq_len, head_dim]
    value,              # [batch, num_heads, kv_seq_len, head_dim]
    attn_mask=mask,     # Optional attention mask
    is_causal=True,     # For causal (autoregressive) attention
)
```

This handles:
- Scaling by 1/sqrt(head_dim)
- Applying attention mask
- Softmax over attention scores
- Matrix multiplication with values

### 5. **Mathematical Operations in ttnn**

Common operations:
```python
# Element-wise operations
ttnn.mul(a, b)        # Element-wise multiplication
ttnn.add(a, b)        # Element-wise addition
ttnn.pow(x, 2)        # Power

# Reductions
ttnn.mean(x, dim=-1, keepdim=True)  # Mean reduction
ttnn.sum(x, dim=-1, keepdim=True)   # Sum reduction

# Other
ttnn.rsqrt(x)         # Reciprocal square root (1/sqrt(x))
```

### 6. **Device Management**

```python
# Open device
device = ttnn.open_device(device_id=0)

# Use device in operations
tensor_tt = ttnn.from_torch(tensor, device=device)

# Close device when done
ttnn.close_device(device)
```

**Important**: For devices that get into a bad state, use `tt-smi -r` to reset.

## Architecture-Specific Considerations

### RoPE (Rotary Position Embeddings)

**CRITICAL GOTCHA**: HuggingFace vs Meta RoPE Format

- **Meta format** (used in original Llama): Interleaved format where rotary dimensions alternate
- **HuggingFace format**: Non-interleaved format with first half and second half

The HuggingFace model uses HuggingFace format, so we must:
1. **NOT** swizzle/convert weights like some existing implementations do
2. Split query/key into two halves along head_dim
3. Apply rotation: `[q1*cos - q2*sin, q1*sin + q2*cos]`

```python
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # Split into halves
    q_half_dim = q.shape[-1] // 2
    q1, q2 = q[..., :q_half_dim], q[..., q_half_dim:]
    k1, k2 = k[..., :q_half_dim], k[..., q_half_dim:]

    # Apply rotation (HuggingFace format)
    q_embed = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q_embed, k_embed
```

### RMSNorm (Root Mean Square Layer Normalization)

Llama uses RMSNorm instead of LayerNorm:

```python
# Formula: x * rsqrt(mean(x^2) + eps) * weight
variance = ttnn.pow(hidden_states, 2)
variance = ttnn.mean(variance, dim=-1, keepdim=True)
hidden_states = hidden_states * ttnn.rsqrt(variance + eps)
hidden_states = ttnn.mul(hidden_states, weight)
```

### Grouped-Query Attention (GQA)

Llama 3.2 1B uses GQA where key/value heads are fewer than query heads:
- Attention heads: 32
- KV heads: 8
- Groups: 4 heads per KV head

Implementation requires repeating KV states:
```python
key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)
```

### SwiGLU MLP

Llama uses SwiGLU activation instead of standard MLP:

```python
gate = silu(gate_proj(x))
up = up_proj(x)
hidden = gate * up  # Element-wise multiplication
output = down_proj(hidden)
```

## HuggingFace Integration

### Making the Model Compatible with generate()

To use HuggingFace's `generate()` function, implement these methods:

1. **forward()**: Standard forward pass returning `CausalLMOutputWithPast`
2. **prepare_inputs_for_generation()**: Prepares inputs for each generation step
3. **_reorder_cache()**: Reorders KV cache for beam search
4. **can_generate()**: Returns True to indicate generation capability

Key attributes needed:
```python
self.config.is_encoder_decoder = False
self.config.model_type = "llama"
self.main_input_name = "input_ids"
```

### Cache Management for Generation

The model must handle KV caching properly:

**Prefill** (first forward pass):
- Input: Full prompt sequence
- `past_key_values=None`
- Computes attention over full sequence
- Returns cache for all positions

**Decode** (subsequent passes):
- Input: Single token
- `past_key_values=<previous cache>`
- Only computes attention for new token
- Appends to cache

Implementation:
```python
if past_key_value is not None:
    key_states = torch.cat([past_key_value[0], key_states], dim=2)
    value_states = torch.cat([past_key_value[1], value_states], dim=2)
```

## Implementation Strategy

### Hybrid Approach: PyTorch for Control Flow, ttnn for Compute

The implementation uses a hybrid strategy:

1. **ttnn for compute-heavy operations**:
   - Linear layers (matrix multiplication)
   - Attention computation
   - Activation functions
   - Normalization

2. **PyTorch for control flow**:
   - Tensor reshaping and views
   - Concatenation and slicing
   - Conditional logic
   - Embedding lookups

This approach provides:
- Clean, readable code
- Easy debugging
- Flexibility to optimize incrementally
- Good performance where it matters

### Development Workflow

1. **Start with structure**: Implement the overall model architecture
2. **Use reference model**: Load pretrained weights from HuggingFace
3. **Incremental conversion**: Convert operations to ttnn one at a time
4. **Test continuously**: Verify each component works correctly
5. **Debug by bisection**: Replace suspected broken parts with reference implementations

## Debugging Techniques

### 1. Component Isolation

If generation produces bad output, systematically replace parts with reference:

```python
# Example: Test if MLP is the problem
class HybridLayer(nn.Module):
    def __init__(self, ttnn_layer, hf_layer):
        self.ttnn_attn = ttnn_layer.self_attn
        self.hf_mlp = hf_layer.mlp  # Use reference MLP

    def forward(self, x, ...):
        # ttnn attention
        x = self.ttnn_attn(x, ...)
        # Reference MLP with tensor conversion
        x_torch = ttnn.to_torch(x)
        x_torch = self.hf_mlp(x_torch)
        x = ttnn.from_torch(x_torch)
        return x
```

This is faster than checking PCC (Pearson Correlation Coefficient) layer by layer.

### 2. Shape Verification

Print shapes liberally during development:
```python
print(f"Query shape: {query.shape}")  # Should be [batch, num_heads, seq_len, head_dim]
print(f"Key shape: {key.shape}")      # Should be [batch, num_kv_heads, kv_seq_len, head_dim]
```

### 3. Numerical Validation

For critical operations, compare against reference:
```python
# Compare ttnn output vs PyTorch reference
output_tt = ttnn.to_torch(ttnn_operation(input_tt))
output_ref = reference_operation(input_torch)
diff = (output_tt - output_ref).abs().max()
print(f"Max difference: {diff}")
```

### 4. Device Reset

If the device gets stuck or produces garbage:
```bash
tt-smi -r  # Reset the device
```

## Configuration Parameters (Llama 3.2 1B)

Key configuration values for reference:
```python
hidden_size = 2048
intermediate_size = 8192
num_hidden_layers = 16
num_attention_heads = 32
num_key_value_heads = 8
max_position_embeddings = 131072
vocab_size = 128256
rms_norm_eps = 1e-5
rope_theta = 500000.0
```

## Performance Considerations

### Memory Layout

For this initial implementation:
- **DRAM interleaved** is fine (no sharding needed)
- Focus on functionality first, optimization later

### Batch Size

Start with batch_size=1 for simplicity:
- Easier to debug
- Simpler shape management
- Can optimize for batching later

### Mixed Precision

The implementation handles dtype conversion automatically:
- Model weights can be in different dtypes
- ttnn operations may use optimal precision internally

## Common Pitfalls

1. **Weight transpose**: Forgetting to transpose PyTorch Linear weights for ttnn
2. **RoPE format**: Using Meta-style RoPE for HuggingFace models
3. **Attention mask**: Not properly handling 4D attention masks
4. **Position IDs**: Incorrect position IDs for decode phase with cache
5. **Cache concatenation**: Not properly concatenating KV cache along sequence dimension
6. **Device management**: Not closing device or recovering from bad state

## Testing Strategy

### 1. Reference Comparison

Test against the HuggingFace reference model:
```python
# Load both models
reference_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
ttnn_model = TtnnLlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device=device)

# Generate with both
ref_output = reference_model.generate(inputs, max_new_tokens=20)
ttnn_output = ttnn_model.generate(inputs, max_new_tokens=20)

# Compare
print("Reference:", tokenizer.decode(ref_output[0]))
print("ttnn:", tokenizer.decode(ttnn_output[0]))
```

### 2. Sequence Continuation Test

Use the prompt "1 2 3 4 5 6 7 8 9 10 11 12":
- Model should continue with "13 14 15..."
- Simple pattern that's easy to verify
- Tests both understanding and generation

### 3. Prefill vs Decode

Test both phases:
- **Prefill**: Long input sequence (e.g., 12 tokens)
- **Decode**: Generates one token at a time with cache

## Files Structure

```
llama32_1b.py               # Main implementation
├── TtnnRMSNorm            # RMS normalization
├── TtnnLlamaAttention     # Self-attention with RoPE
├── TtnnLlamaMLP           # SwiGLU feed-forward
├── TtnnLlamaDecoderLayer  # Transformer block
├── TtnnLlamaModel         # Full model
├── TtnnLlamaForCausalLM   # Causal LM head + HF integration
└── main()                  # CLI interface
```

## Running the Model

Basic usage:
```bash
python llama32_1b.py --prompt "1 2 3 4 5 6 7 8 9 10 11 12" --max_new_tokens 20
```

Options:
- `--prompt`: Input text to generate from
- `--max_new_tokens`: Number of tokens to generate (default: 20)
- `--model_name`: HuggingFace model name (default: meta-llama/Llama-3.2-1B)

## Future Optimizations

Once the basic implementation works, consider:

1. **Multi-device support**: Distribute layers across devices
2. **Tensor sharding**: Shard tensors for parallel computation
3. **Weight preprocessing**: Convert all weights to ttnn format upfront
4. **Fused operations**: Combine multiple operations into custom kernels
5. **Optimized layouts**: Use L1/SRAM instead of DRAM where beneficial
6. **Batching**: Support batch_size > 1 efficiently

## Key Takeaways

1. **Start simple**: Functional > optimal. Get it working, then optimize.
2. **Know the quirks**: ttnn.linear weight format is different from PyTorch.
3. **Hybrid is OK**: Mix PyTorch and ttnn as needed for clarity.
4. **Use built-ins**: ttnn.transformer functions are your friend.
5. **RoPE matters**: HuggingFace format != Meta format. Don't convert weights.
6. **Test incrementally**: Binary search for bugs with reference implementations.
7. **HuggingFace integration**: Worth the effort for clean generation interface.
8. **Device management**: Remember to open/close devices and reset if needed.

## Resources

- ttnn documentation: Check `ttnn` module docstrings and examples
- HuggingFace transformers: Reference implementations
- Llama paper: For architectural details
- This implementation: A clean starting point for learning ttnn

## Conclusion

This implementation demonstrates that bringing up a model in ttnn can be straightforward when you:
- Understand the key differences (like linear weight format)
- Use a hybrid approach for readability
- Leverage built-in functions (like scaled_dot_product_attention)
- Maintain compatibility with existing ecosystems (HuggingFace)

The result is clean, educational code that serves as a foundation for more advanced optimizations while remaining accessible to developers new to ttnn.
