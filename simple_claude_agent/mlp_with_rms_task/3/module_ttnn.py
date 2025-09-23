import ttnn

class LlamaMLP:
    """
    Multi-Layer Perceptron (MLP) for LLaMA model ported to ttnn
    """

    def __init__(self, config, torch_params=None, device=None):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = device

        # Load parameters from CPU torch tensors if provided
        if torch_params is not None:
            param_dict = dict(torch_params)

            # Convert linear layer weights and biases to ttnn tensors
            gate_weight = param_dict['gate_proj.weight']
            up_weight = param_dict['up_proj.weight']
            down_weight = param_dict['down_proj.weight']

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
        else:
            # Initialize with random weights if no parameters provided
            self.gate_weight = ttnn.ones(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_weight = ttnn.ones(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_weight = ttnn.ones(
                (self.hidden_size, self.intermediate_size),
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
        # Gate projection: x @ gate_weight.T + gate_bias
        gate_proj = ttnn.linear(
            x,
            self.gate_weight,
            bias=self.gate_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Up projection: x @ up_weight.T + up_bias
        up_proj = ttnn.linear(
            x,
            self.up_weight,
            bias=self.up_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Apply SiLU activation to gate projection
        gate_activated = ttnn.silu(gate_proj)

        # Element-wise multiplication: gate_activated * up_proj
        intermediate = ttnn.multiply(
            gate_activated,
            up_proj,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Down projection: intermediate @ down_weight.T + down_bias
        output = ttnn.linear(
            intermediate,
            self.down_weight,
            bias=self.down_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return output