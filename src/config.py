"""
Qwen3 Model Configuration
Author: Bound
Date: May 1, 2025
Version: 1.0
"""

import torch

class Qwen3Config:
    def __init__(
        self,
        vocab_size=151936,  
        hidden_size=1024,
        intermediate_size=3072,
        head_dim=128,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        torch_type=torch.bfloat16,
        eos_token_id=151645
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim=head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.torch_type = torch_type
        self.eos_token_id = eos_token_id