import ttnn

# Inline activation function mapping - we only need SiLU for Llama
ACT2FN = {'silu': ttnn.silu, 'swish': ttnn.silu}

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

            # Convert weight matrices to ttnn tensors
            # Note: ttnn.linear expects (A,B) inputs and (B,C) weights,
            # while torch.nn.functional.linear expects (A,B) inputs and (C,B) weights
            # So we need to transpose the PyTorch weights
            gate_weight = param_dict['gate_proj.weight'].T  # Transpose for ttnn
            up_weight = param_dict['up_proj.weight'].T      # Transpose for ttnn
            down_weight = param_dict['down_proj.weight'].T  # Transpose for ttnn

            self.gate_proj_weight = ttnn.from_torch(
                gate_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_proj_weight = ttnn.from_torch(
                up_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_proj_weight = ttnn.from_torch(
                down_weight,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Handle biases if they exist
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
            # Initialize with random weights if no parameters provided
            self.gate_proj_weight = ttnn.zeros(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.up_proj_weight = ttnn.zeros(
                (self.hidden_size, self.intermediate_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.down_proj_weight = ttnn.zeros(
                (self.intermediate_size, self.hidden_size),
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            self.gate_proj_bias = None
            self.up_proj_bias = None
            self.down_proj_bias = None

        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        """
        # Gate projection
        gate_output = ttnn.linear(x, self.gate_proj_weight, bias=self.gate_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Up projection
        up_output = ttnn.linear(x, self.up_proj_weight, bias=self.up_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Apply activation function to gate output
        gate_activated = self.act_fn(gate_output)

        # Element-wise multiplication of activated gate and up projections
        gated_up = ttnn.multiply(gate_activated, up_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection
        down_proj = ttnn.linear(gated_up, self.down_proj_weight, bias=self.down_proj_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return down_proj