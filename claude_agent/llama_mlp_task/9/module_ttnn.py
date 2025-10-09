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

            # Extract weights for each linear layer
            gate_weight = param_dict['gate_proj.weight']
            up_weight = param_dict['up_proj.weight']
            down_weight = param_dict['down_proj.weight']

            # Convert weights to ttnn tensors
            # Note: ttnn.linear expects (A,B) inputs and (B,C) weights,
            # while torch uses (A,B) inputs and (C,B) weights, so we need to transpose
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

        # Get activation function
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, x):
        """
        Forward pass - expects x to already be a ttnn tensor on device
        """
        # Gate projection: x @ gate_weight + gate_bias (optional)
        gate_proj = ttnn.linear(
            x,
            self.gate_weight,
            bias=self.gate_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Up projection: x @ up_weight + up_bias (optional)
        up_proj = ttnn.linear(
            x,
            self.up_weight,
            bias=self.up_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Apply activation to gate projection
        gate_activated = self.act_fn(gate_proj)

        # Element-wise multiplication
        intermediate = ttnn.multiply(gate_activated, up_proj, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Down projection: intermediate @ down_weight + down_bias (optional)
        down_proj = ttnn.linear(
            intermediate,
            self.down_weight,
            bias=self.down_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        return down_proj