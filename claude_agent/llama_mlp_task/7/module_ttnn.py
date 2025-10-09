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
            # Note: ttnn.linear expects (B,C) weights, so we need to transpose
            self.gate_weight = ttnn.from_torch(
                gate_weight.T,  # Transpose for ttnn convention
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_weight = ttnn.from_torch(
                up_weight.T,  # Transpose for ttnn convention
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_weight = ttnn.from_torch(
                down_weight.T,  # Transpose for ttnn convention
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Handle bias if present
            self.has_bias = config.mlp_bias
            if self.has_bias:
                gate_bias = param_dict.get('gate_proj.bias')
                up_bias = param_dict.get('up_proj.bias')
                down_bias = param_dict.get('down_proj.bias')

                if gate_bias is not None:
                    self.gate_bias = ttnn.from_torch(
                        gate_bias,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                if up_bias is not None:
                    self.up_bias = ttnn.from_torch(
                        up_bias,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                if down_bias is not None:
                    self.down_bias = ttnn.from_torch(
                        down_bias,
                        device=device,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
        else:
            # Initialize with random weights if no parameters provided
            self.gate_weight = ttnn.random_normal(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_weight = ttnn.random_normal(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_weight = ttnn.random_normal(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.has_bias = False

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        """
        # Gate projection: x @ gate_weight + gate_bias (if present)
        gate_output = ttnn.linear(x, self.gate_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.has_bias and hasattr(self, 'gate_bias'):
            gate_output = ttnn.add(gate_output, self.gate_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Up projection: x @ up_weight + up_bias (if present)
        up_output = ttnn.linear(x, self.up_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.has_bias and hasattr(self, 'up_bias'):
            up_output = ttnn.add(up_output, self.up_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply SiLU activation to gate output
        gate_activated = ttnn.silu(gate_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Element-wise multiplication: silu(gate) * up
        intermediate = ttnn.multiply(gate_activated, up_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection: intermediate @ down_weight + down_bias (if present)
        output = ttnn.linear(intermediate, self.down_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.has_bias and hasattr(self, 'down_bias'):
            output = ttnn.add(output, self.down_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return output