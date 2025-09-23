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

            # Convert gate projection weights and bias to ttnn tensors
            gate_weight = param_dict['gate_proj.weight']
            self.gate_proj_weight = ttnn.from_torch(
                gate_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            if config.mlp_bias and 'gate_proj.bias' in param_dict:
                gate_bias = param_dict['gate_proj.bias']
                self.gate_proj_bias = ttnn.from_torch(
                    gate_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                self.gate_proj_bias = None

            # Convert up projection weights and bias to ttnn tensors
            up_weight = param_dict['up_proj.weight']
            self.up_proj_weight = ttnn.from_torch(
                up_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            if config.mlp_bias and 'up_proj.bias' in param_dict:
                up_bias = param_dict['up_proj.bias']
                self.up_proj_bias = ttnn.from_torch(
                    up_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                self.up_proj_bias = None

            # Convert down projection weights and bias to ttnn tensors
            down_weight = param_dict['down_proj.weight']
            self.down_proj_weight = ttnn.from_torch(
                down_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            if config.mlp_bias and 'down_proj.bias' in param_dict:
                down_bias = param_dict['down_proj.bias']
                self.down_proj_bias = ttnn.from_torch(
                    down_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                self.down_proj_bias = None
        else:
            # Initialize with random weights if no parameters provided
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
        # Gate projection: linear transformation
        gate_out = ttnn.linear(
            x,
            self.gate_proj_weight,
            bias=self.gate_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Apply SiLU activation to gate output
        gate_activated = ttnn.silu(gate_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Up projection: linear transformation
        up_out = ttnn.linear(
            x,
            self.up_proj_weight,
            bias=self.up_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Element-wise multiplication of activated gate and up projection
        gated_up = ttnn.multiply(gate_activated, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection: final linear transformation
        down_proj = ttnn.linear(
            gated_up,
            self.down_proj_weight,
            bias=self.down_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return down_proj