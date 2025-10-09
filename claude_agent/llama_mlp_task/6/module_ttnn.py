import ttnn


class LlamaMLP:
    """
    LlamaMLP module ported to ttnn
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
            gate_weight = param_dict['gate_proj.weight']  # (intermediate_size, hidden_size)
            up_weight = param_dict['up_proj.weight']      # (intermediate_size, hidden_size)
            down_weight = param_dict['down_proj.weight']  # (hidden_size, intermediate_size)

            # Convert weights to ttnn tensors
            # Note: ttnn.linear expects weights in (input_size, output_size) format
            # PyTorch linear weights are (output_size, input_size), so we need to transpose
            self.gate_weight = ttnn.from_torch(
                gate_weight.transpose(-1, -2),  # (hidden_size, intermediate_size)
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_weight = ttnn.from_torch(
                up_weight.transpose(-1, -2),    # (hidden_size, intermediate_size)
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_weight = ttnn.from_torch(
                down_weight.transpose(-1, -2),  # (intermediate_size, hidden_size)
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Handle bias if present
            self.gate_bias = None
            self.up_bias = None
            self.down_bias = None

            if config.mlp_bias:
                if 'gate_proj.bias' in param_dict:
                    self.gate_bias = ttnn.from_torch(
                        param_dict['gate_proj.bias'],
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                if 'up_proj.bias' in param_dict:
                    self.up_bias = ttnn.from_torch(
                        param_dict['up_proj.bias'],
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                if 'down_proj.bias' in param_dict:
                    self.down_bias = ttnn.from_torch(
                        param_dict['down_proj.bias'],
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
        else:
            # Initialize with random weights if no parameters provided
            self.gate_weight = ttnn.zeros(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.up_weight = ttnn.zeros(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            self.down_weight = ttnn.zeros(
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
        Implements: down_proj(silu(gate_proj(x)) * up_proj(x))
        """
        # Gate projection: x @ gate_weight
        gate_out = ttnn.linear(x, self.gate_weight, bias=self.gate_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Up projection: x @ up_weight
        up_out = ttnn.linear(x, self.up_weight, bias=self.up_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply SiLU activation to gate output: silu(gate_proj(x))
        gate_activated = ttnn.silu(gate_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Element-wise multiplication: silu(gate_proj(x)) * up_proj(x)
        intermediate = ttnn.multiply(gate_activated, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection: intermediate @ down_weight
        output = ttnn.linear(intermediate, self.down_weight, bias=self.down_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return output