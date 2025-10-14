import ttnn

HIGH_FIDELITY_CONFIG  = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
class LlamaMLP:
    """
    Multi-Layer Perceptron (MLP) for LLaMA model ported to ttnn
    """

    def __init__(self, config, torch_params=None, device=None):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = device

        # Load parameters from CPU torch tensors
        if torch_params is not None:
            param_dict = dict(torch_params)
            
            # Load gate projection weight (transpose for TT)
            gate_weight = param_dict['gate_proj.weight'].T.contiguous()
            self.gate_proj_weight = ttnn.from_torch(
                gate_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            # Load up projection weight (transpose for TT)
            up_weight = param_dict['up_proj.weight'].T.contiguous()
            self.up_proj_weight = ttnn.from_torch(
                up_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            # Load down projection weight (transpose for TT)
            down_weight = param_dict['down_proj.weight'].T.contiguous()
            self.down_proj_weight = ttnn.from_torch(
                down_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            
            # Handle bias if present (though typically False for LLaMA)
            if config.mlp_bias:
                if 'gate_proj.bias' in param_dict:
                    gate_bias = param_dict['gate_proj.bias'].reshape(1, -1)
                    self.gate_proj_bias = ttnn.from_torch(
                        gate_bias,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                else:
                    self.gate_proj_bias = None
                    
                if 'up_proj.bias' in param_dict:
                    up_bias = param_dict['up_proj.bias'].reshape(1, -1)
                    self.up_proj_bias = ttnn.from_torch(
                        up_bias,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                else:
                    self.up_proj_bias = None
                    
                if 'down_proj.bias' in param_dict:
                    down_bias = param_dict['down_proj.bias'].reshape(1, -1)
                    self.down_proj_bias = ttnn.from_torch(
                        down_bias,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                else:
                    self.down_proj_bias = None
            else:
                self.gate_proj_bias = None
                self.up_proj_bias = None
                self.down_proj_bias = None

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        MLP computation: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        """
        # Gate projection
        gate_out = ttnn.linear(
            x,
            self.gate_proj_weight,
            bias=self.gate_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=HIGH_FIDELITY_CONFIG
        )
        
        # Up projection  
        up_out = ttnn.linear(
            x,
            self.up_proj_weight,
            bias=self.up_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=HIGH_FIDELITY_CONFIG
        )
        
        # Apply SiLU activation to gate output
        gate_activated = ttnn.silu(gate_out)
        
        # Element-wise multiplication
        intermediate = ttnn.multiply(gate_activated, up_out)
        
        # Down projection
        output = ttnn.linear(
            intermediate,
            self.down_proj_weight,
            bias=self.down_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=HIGH_FIDELITY_CONFIG
        )
        
        return output