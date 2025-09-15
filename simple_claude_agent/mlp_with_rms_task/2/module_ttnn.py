import ttnn


class LlamaMLP:
    """
    Multi-Layer Perceptron layer for LLaMA model ported to ttnn
    """

    def __init__(self, config, torch_params=None, device=None):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = device

        # Load parameters from CPU torch tensors
        if torch_params is not None:
            param_dict = dict(torch_params)

            # Convert gate_proj weight to ttnn tensor
            gate_proj_weight = param_dict['gate_proj.weight']
            self.gate_proj_weight = ttnn.from_torch(
                gate_proj_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Convert up_proj weight to ttnn tensor
            up_proj_weight = param_dict['up_proj.weight']
            self.up_proj_weight = ttnn.from_torch(
                up_proj_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Convert down_proj weight to ttnn tensor
            down_proj_weight = param_dict['down_proj.weight']
            self.down_proj_weight = ttnn.from_torch(
                down_proj_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Handle bias if present
            if config.mlp_bias:
                gate_proj_bias = param_dict['gate_proj.bias']
                self.gate_proj_bias = ttnn.from_torch(
                    gate_proj_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                up_proj_bias = param_dict['up_proj.bias']
                self.up_proj_bias = ttnn.from_torch(
                    up_proj_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

                down_proj_bias = param_dict['down_proj.bias']
                self.down_proj_bias = ttnn.from_torch(
                    down_proj_bias,
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                self.gate_proj_bias = None
                self.up_proj_bias = None
                self.down_proj_bias = None
        else:
            # Initialize with random weights if no parameters provided
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
        gate_output = ttnn.matmul(x, self.gate_proj_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.gate_proj_bias is not None:
            gate_output = ttnn.add(gate_output, self.gate_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Up projection: x @ up_proj_weight.T + up_proj_bias
        up_output = ttnn.matmul(x, self.up_proj_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.up_proj_bias is not None:
            up_output = ttnn.add(up_output, self.up_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply SiLU activation to gate projection: gate_output * sigmoid(gate_output)
        gate_sigmoid = ttnn.sigmoid(gate_output)
        gate_silu = ttnn.multiply(gate_output, gate_sigmoid, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Element-wise multiplication: silu(gate) * up
        intermediate = ttnn.multiply(gate_silu, up_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection: intermediate @ down_proj_weight.T + down_proj_bias
        down_output = ttnn.matmul(intermediate, self.down_proj_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.down_proj_bias is not None:
            down_output = ttnn.add(down_output, self.down_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return down_output