import ttnn

class LlamaRMSNorm:
    """
    RMS Normalization layer for LLaMA model ported to ttnn
    """

    def __init__(self, config, torch_params=None, device=None):
        self.hidden_size = config.hidden_size
        self.variance_epsilon = config.rms_norm_eps
        self.device = device

        # Load parameters from CPU torch tensors
        if torch_params is not None:
            param_dict = dict(torch_params)
            weight = param_dict['weight']
            
            # Convert weight to ttnn tensor
            self.weight = ttnn.from_torch(
                weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            # Initialize with ones if no parameters provided
            self.weight = ttnn.ones(
                (self.hidden_size,),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

    def __call__(self, hidden_states):
        """
        Forward pass - expects hidden_states to already be a ttnn tensor on device
        """
        # Calculate variance: mean of squares along last dimension
        squared = ttnn.pow(hidden_states, 2.0)
        variance = ttnn.mean(squared, dim=-1, keepdim=True)
        
        # Add epsilon and take reciprocal square root
        variance_eps = ttnn.add(variance, self.variance_epsilon)
        inv_sqrt = ttnn.rsqrt(variance_eps)
        
        # Normalize hidden states
        normalized = ttnn.multiply(hidden_states, inv_sqrt)
        
        # Apply weight scaling
        output = ttnn.multiply(normalized, self.weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        return output