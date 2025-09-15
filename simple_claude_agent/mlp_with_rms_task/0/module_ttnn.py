import ttnn

class LlamaMLP:
    """
    MLP layer for LLaMA model ported to ttnn
    """

    def __init__(self, config, torch_params=None, device=None):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = device

        # Load parameters from CPU torch tensors
        if torch_params is not None:
            param_dict = dict(torch_params)
            
            # Convert gate_proj weight to ttnn tensor
            gate_weight = param_dict['gate_proj.weight']
            self.gate_proj_weight = ttnn.from_torch(
                gate_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            # Convert up_proj weight to ttnn tensor
            up_weight = param_dict['up_proj.weight']
            self.up_proj_weight = ttnn.from_torch(
                up_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            # Convert down_proj weight to ttnn tensor
            down_weight = param_dict['down_proj.weight']
            self.down_proj_weight = ttnn.from_torch(
                down_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            # Handle bias if present
            if config.mlp_bias:
                gate_bias = param_dict['gate_proj.bias']
                self.gate_proj_bias = ttnn.from_torch(
                    gate_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                
                up_bias = param_dict['up_proj.bias']
                self.up_proj_bias = ttnn.from_torch(
                    up_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                
                down_bias = param_dict['down_proj.bias']
                self.down_proj_bias = ttnn.from_torch(
                    down_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                self.gate_proj_bias = None
                self.up_proj_bias = None
                self.down_proj_bias = None
        else:
            # Initialize weights if no parameters provided
            self.gate_proj_weight = ttnn.zeros(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.up_proj_weight = ttnn.zeros(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.down_proj_weight = ttnn.zeros(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.gate_proj_bias = None
            self.up_proj_bias = None
            self.down_proj_bias = None

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        """
        # Gate projection: Linear(hidden_size, intermediate_size)
        gate_out = ttnn.linear(x, self.gate_proj_weight, bias=self.gate_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Up projection: Linear(hidden_size, intermediate_size)
        up_out = ttnn.linear(x, self.up_proj_weight, bias=self.up_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Apply SiLU activation to gate output
        gate_activated = ttnn.silu(gate_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Element-wise multiplication of activated gate and up projections
        intermediate = ttnn.multiply(gate_activated, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Down projection: Linear(intermediate_size, hidden_size)
        output = ttnn.linear(intermediate, self.down_proj_weight, bias=self.down_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        return output