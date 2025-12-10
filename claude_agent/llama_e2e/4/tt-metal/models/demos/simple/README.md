# Simple Model Demos

This directory contains simple, educational implementations of models in ttnn. These implementations prioritize clarity and ease of understanding over performance optimization.

## Available Models

### Llama 3.2 1B (`llama32_1b.py`)

A straightforward implementation of Llama 3.2 1B that demonstrates:
- Loading HuggingFace models and converting to ttnn
- Using ttnn operations for transformer architecture
- Integration with HuggingFace's `generate()` function
- Both prefill and decode modes

**Usage:**
```bash
python llama32_1b.py "Your prompt here"

# Example: Test number continuation
python llama32_1b.py "1 2 3 4 5 6 7 8 9 10 11 12"
```

**Requirements:**
- HuggingFace transformers library
- Access to meta-llama/Llama-3.2-1B model
- ttnn device (Wormhole or Blackhole)

**Features:**
- ✅ Clean, readable code
- ✅ HuggingFace integration
- ✅ Proper RoPE handling (HuggingFace format)
- ✅ Grouped Query Attention (GQA)
- ✅ SiLU-gated MLP
- ✅ RMSNorm

**Not Included (by design for simplicity):**
- Multi-device support
- Advanced sharding
- Optimized KV cache
- Performance tuning

See `MODEL_BRINGUP.md` in the repo root for detailed implementation notes and learnings.

## Philosophy

These demos are designed to:
1. **Teach:** Show how to use ttnn operations directly
2. **Inspire:** Serve as starting points for your own implementations
3. **Debug:** Provide reference implementations for comparison

For production-quality, optimized implementations, see the main `models/` directory.
