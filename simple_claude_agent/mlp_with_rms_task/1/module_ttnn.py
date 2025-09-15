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
            
            # Convert weights to ttnn tensors
            self.gate_proj_weight = ttnn.from_torch(
                param_dict['gate_proj.weight'],
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            self.up_proj_weight = ttnn.from_torch(
                param_dict['up_proj.weight'],
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            self.down_proj_weight = ttnn.from_torch(
                param_dict['down_proj.weight'],
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            # Handle bias if present
            self.gate_proj_bias = None
            self.up_proj_bias = None
            self.down_proj_bias = None
            
            if config.mlp_bias:
                if 'gate_proj.bias' in param_dict:
                    self.gate_proj_bias = ttnn.from_torch(
                        param_dict['gate_proj.bias'],
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                
                if 'up_proj.bias' in param_dict:
                    self.up_proj_bias = ttnn.from_torch(
                        param_dict['up_proj.bias'],
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                
                if 'down_proj.bias' in param_dict:
                    self.down_proj_bias = ttnn.from_torch(
                        param_dict['down_proj.bias'],
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        """
        # Gate projection: gate_proj(x)
        gate_out = ttnn.linear(
            x, 
            self.gate_proj_weight,
            bias=self.gate_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        # Apply SiLU activation to gate output
        gate_activated = ttnn.silu(gate_out)
        
        # Up projection: up_proj(x)
        up_out = ttnn.linear(
            x,
            self.up_proj_weight,
            bias=self.up_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        # Element-wise multiplication: gate_activated * up_out
        gated = ttnn.multiply(gate_activated, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        
        # Down projection: down_proj(gated)
        down_proj = ttnn.linear(
            gated,
            self.down_proj_weight,
            bias=self.down_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        
        return down_proj