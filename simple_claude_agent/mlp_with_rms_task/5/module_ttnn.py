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

            # Convert linear layer weights to ttnn tensors
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
            if config.mlp_bias:
                self.gate_proj_bias = ttnn.from_torch(
                    param_dict['gate_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                self.up_proj_bias = ttnn.from_torch(
                    param_dict['up_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                self.down_proj_bias = ttnn.from_torch(
                    param_dict['down_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                self.gate_proj_bias = None
                self.up_proj_bias = None
                self.down_proj_bias = None
        else:
            # Initialize with random values if no parameters provided
            self.gate_proj_weight = ttnn.ones(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_proj_weight = ttnn.ones(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_proj_weight = ttnn.ones(
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
        # Gate projection: x @ gate_proj_weight.T + gate_proj_bias
        gate_proj = ttnn.linear(
            x,
            self.gate_proj_weight,
            bias=self.gate_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Up projection: x @ up_proj_weight.T + up_proj_bias
        up_proj = ttnn.linear(
            x,
            self.up_proj_weight,
            bias=self.up_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Apply SiLU activation to gate projection
        gate_proj_activated = ttnn.silu(gate_proj)

        # Element-wise multiplication of activated gate projection and up projection
        intermediate = ttnn.multiply(
            gate_proj_activated,
            up_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Down projection: intermediate @ down_proj_weight.T + down_proj_bias
        down_proj = ttnn.linear(
            intermediate,
            self.down_proj_weight,
            bias=self.down_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return down_proj