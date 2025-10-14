"""
RMS Normalization layer for LLaMA model ported to ttnn
"""

import ttnn

HIGH_FIDELITY_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

class LlamaRMSNorm:
    """
    LlamaRMSNorm is equivalent to T5LayerNorm ported to ttnn
    """

    def __init__(self, hidden_size, eps=1e-6, torch_params=None, device=None):
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.device = device

        # Load parameters from CPU torch tensors
        if torch_params is not None:
            param_dict = dict(torch_params)

            # Load weight parameter
            weight = param_dict['weight']
            self.weight = ttnn.from_torch(
                weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            # Initialize with ones if no parameters provided
            import torch
            weight_tensor = torch.ones(hidden_size)
            self.weight = ttnn.from_torch(
                weight_tensor,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

    def __call__(self, hidden_states):
        """
        Forward pass - expects hidden_states to already be a ttnn tensor on device
        """
        # Convert to float32 for computation precision
        input_dtype = hidden_states.dtype
        if input_dtype != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        # Compute variance: hidden_states.pow(2).mean(-1, keepdim=True)
        squared = ttnn.pow(hidden_states, 2.0)
        variance = ttnn.mean(squared, dim=-1, keepdim=True)

        # Add epsilon and compute rsqrt
        variance_eps = ttnn.add(variance, self.variance_epsilon)
        rsqrt_var = ttnn.rsqrt(variance_eps)

        # Normalize: hidden_states * rsqrt(variance + eps)
        normalized = ttnn.multiply(hidden_states, rsqrt_var)

        # Apply weight scaling
        output = ttnn.multiply(self.weight, normalized)

        # Convert back to original dtype if needed
        if input_dtype != ttnn.bfloat16:
            output = ttnn.typecast(output, input_dtype)

        return output

    def extra_repr(self):
        return f"({self.hidden_size},), eps={self.variance_epsilon}"