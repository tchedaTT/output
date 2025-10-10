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
            # Note: ttnn.linear expects (A,B) inputs and (B,C) weights,
            # in contrast to torch.nn.functional.linear which expects (C,B) weights
            gate_weight = param_dict['gate_proj.weight'].T  # Transpose for ttnn
            up_weight = param_dict['up_proj.weight'].T      # Transpose for ttnn
            down_weight = param_dict['down_proj.weight'].T  # Transpose for ttnn

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
            if config.mlp_bias:
                gate_bias = param_dict['gate_proj.bias']
                up_bias = param_dict['up_proj.bias']
                down_bias = param_dict['down_proj.bias']

                self.gate_bias = ttnn.from_torch(
                    gate_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                self.up_bias = ttnn.from_torch(
                    up_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                self.down_bias = ttnn.from_torch(
                    down_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
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
        gate_activated = ttnn.silu(gate_out)

        # Element-wise multiplication of activated gate and up projections
        intermediate = ttnn.multiply(gate_activated, up_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection
        output = ttnn.linear(intermediate, self.down_weight, bias=self.down_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return output