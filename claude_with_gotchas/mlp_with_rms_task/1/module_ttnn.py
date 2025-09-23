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

            # Extract weights for each linear layer
            gate_weight = param_dict['gate_proj.weight']
            up_weight = param_dict['up_proj.weight']
            down_weight = param_dict['down_proj.weight']

            # Convert weights to ttnn tensors
            # Note: ttnn.linear expects (A,B) inputs and (B,C) weights
            # PyTorch uses (C,B) weights, so we need to transpose
            self.gate_weight = ttnn.from_torch(
                gate_weight.transpose(0, 1),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_weight = ttnn.from_torch(
                up_weight.transpose(0, 1),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_weight = ttnn.from_torch(
                down_weight.transpose(0, 1),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Handle bias if present
            if config.mlp_bias:
                self.gate_bias = ttnn.from_torch(
                    param_dict['gate_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                self.up_bias = ttnn.from_torch(
                    param_dict['up_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                self.down_bias = ttnn.from_torch(
                    param_dict['down_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                self.gate_bias = None
                self.up_bias = None
                self.down_bias = None
        else:
            # Initialize with random weights if no parameters provided
            self.gate_weight = ttnn.ones(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_weight = ttnn.ones(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_weight = ttnn.ones(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.gate_bias = None
            self.up_bias = None
            self.down_bias = None

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        """
        # Gate projection
        gate_out = ttnn.linear(x, self.gate_weight, bias=self.gate_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Up projection
        up_out = ttnn.linear(x, self.up_weight, bias=self.up_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply SiLU activation to gate output
        gate_activated = ttnn.silu(gate_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Element-wise multiply gate_activated * up_out
        intermediate = ttnn.multiply(gate_activated, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection
        down_proj = ttnn.linear(intermediate, self.down_weight, bias=self.down_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return down_proj