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

            # Convert weight tensors to ttnn format
            # Note: ttnn.linear expects (B,C) weights, so we need to transpose from torch's (C,B)
            gate_weight = param_dict['gate_proj.weight'].T  # Transpose from (intermediate, hidden) to (hidden, intermediate)
            up_weight = param_dict['up_proj.weight'].T      # Transpose from (intermediate, hidden) to (hidden, intermediate)
            down_weight = param_dict['down_proj.weight'].T  # Transpose from (hidden, intermediate) to (intermediate, hidden)

            self.gate_weight = ttnn.from_torch(
                gate_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_weight = ttnn.from_torch(
                up_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_weight = ttnn.from_torch(
                down_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Handle bias if present
            self.has_bias = config.mlp_bias
            if self.has_bias:
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

            self.has_bias = False

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        Implements: down_proj(act_fn(gate_proj(x)) * up_proj(x))
        """
        # Gate projection
        gate_out = ttnn.linear(x, self.gate_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.has_bias:
            gate_out = ttnn.add(gate_out, self.gate_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Up projection
        up_out = ttnn.linear(x, self.up_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.has_bias:
            up_out = ttnn.add(up_out, self.up_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply SiLU activation to gate output
        gate_activated = ttnn.silu(gate_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Element-wise multiplication
        gated_up = ttnn.multiply(gate_activated, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection
        output = ttnn.linear(gated_up, self.down_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.has_bias:
            output = ttnn.add(output, self.down_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return output