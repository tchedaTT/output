import ttnn

# Inline activation function mapping - we only need SiLU for Llama
# Note: ttnn uses silu directly as an operation
ACT2FN = {'silu': ttnn.silu, 'swish': ttnn.silu}


class LlamaMLP:
    """
    LLaMA MLP layer ported to ttnn
    """

    def __init__(self, config, torch_params=None, device=None):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.device = device

        # Load parameters from CPU torch tensors
        if torch_params is not None:
            param_dict = dict(torch_params)

            # Extract weights for each projection layer
            gate_proj_weight = param_dict['gate_proj.weight']
            up_proj_weight = param_dict['up_proj.weight']
            down_proj_weight = param_dict['down_proj.weight']

            # Convert weights to ttnn tensors
            # Note: ttnn.linear expects (B,C) weights, so we transpose from PyTorch's (C,B)
            self.gate_proj_weight = ttnn.from_torch(
                gate_proj_weight.T,  # Transpose from (intermediate_size, hidden_size) to (hidden_size, intermediate_size)
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_proj_weight = ttnn.from_torch(
                up_proj_weight.T,  # Transpose from (intermediate_size, hidden_size) to (hidden_size, intermediate_size)
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_proj_weight = ttnn.from_torch(
                down_proj_weight.T,  # Transpose from (hidden_size, intermediate_size) to (intermediate_size, hidden_size)
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
                ) if 'gate_proj.bias' in param_dict else None

                self.up_proj_bias = ttnn.from_torch(
                    param_dict['up_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                ) if 'up_proj.bias' in param_dict else None

                self.down_proj_bias = ttnn.from_torch(
                    param_dict['down_proj.bias'],
                    device=device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG
                ) if 'down_proj.bias' in param_dict else None
            else:
                self.gate_proj_bias = None
                self.up_proj_bias = None
                self.down_proj_bias = None
        else:
            # Initialize with random weights if no parameters provided
            # Note: In practice, this would need proper initialization
            self.gate_proj_weight = ttnn.ones(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_proj_weight = ttnn.ones(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_proj_weight = ttnn.ones(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.gate_proj_bias = None
            self.up_proj_bias = None
            self.down_proj_bias = None

        # Set activation function
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        """
        # Gate projection: x @ gate_proj_weight
        gate_output = ttnn.linear(
            x,
            self.gate_proj_weight,
            bias=self.gate_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Apply activation function (SiLU/Swish)
        gate_activated = self.act_fn(gate_output)

        # Up projection: x @ up_proj_weight
        up_output = ttnn.linear(
            x,
            self.up_proj_weight,
            bias=self.up_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Element-wise multiplication of activated gate and up projections
        gated_up = ttnn.multiply(gate_activated, up_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection: gated_up @ down_proj_weight
        down_output = ttnn.linear(
            gated_up,
            self.down_proj_weight,
            bias=self.down_proj_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return down_output