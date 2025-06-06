"""
Qwen3 Model Implementation
Author: Bound
Date: May 14, 2025
Version: 1.0
"""

import torch
import torch.nn as nn
from typing import Tuple
from config import Qwen3Config
'''
  RMSNorm
'''
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor):
        norm_x = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm_x + self.eps)
        return x * self.weight
    
    def __repr__(self):
        return f"{self.__class__.__name__}(({self.hidden_size},), eps={self.eps})"


'''
  RoPE Position Embedding
'''
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

def _compute_inv_freq(config: Qwen3Config) -> torch.Tensor:
    head_dim = config.head_dim
    inv_freq = 1.0 / (
        config.rope_theta ** 
        (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device="cpu", dtype=torch.float32) / head_dim)
    )
    return inv_freq

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.max_seq_len = config.max_position_embeddings
        self.inv_freq = _compute_inv_freq(config)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()                                                  
        
        with torch.autocast(device_type=device.type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() 
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

'''
  MLP
'''
class MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

'''
 Attention with GQA
'''
class KVCache:
    def __init__(self, config: Qwen3Config):
        self.head_dim = config.head_dim
        self.max_seq_len = config.max_position_embeddings
        self.key_cache = torch.zeros(
            (1, config.num_key_value_heads, self.max_seq_len, self.head_dim), 
                dtype=config.torch_type)
        self.value_cache = torch.zeros(
            (1, config.num_key_value_heads, self.max_seq_len, self.head_dim), 
                dtype=config.torch_type)
        self.to_device = False
    
    def update_kv_states(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        is_prefill: bool,
        cache_position: int
    ):
        if cache_position > self.max_seq_len:
            raise ValueError("Current token exceeds maximum sequence length")
        
        if self.to_device == False:
            self.key_cache = self.key_cache.to(key_states.device)
            self.value_cache = self.value_cache.to(value_states.device)
            self.to_device = True

        if is_prefill:
            self.key_cache[:, :, 0:cache_position, :] = key_states
            self.value_cache[:, :, 0:cache_position, :] = value_states
        else :
            self.key_cache[:, :, cache_position-1:cache_position, :] = key_states
            self.value_cache[:, :, cache_position-1:cache_position, :] = value_states
    
    def get_kv_states(
        self,
        cache_position: int
    ) -> Tuple[torch.Tensor, torch.tensor]:
        if cache_position > self.max_seq_len:
            raise ValueError("Current token exceeds maximum sequence length")

        key_states = self.key_cache[:, :, :cache_position].contiguous()
        value_states = self.value_cache[:, :, :cache_position].contiguous()

        return key_states, value_states

def repeat_kv(
        hidden_states: torch.Tensor, 
        n_rep: int
    ) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        is_prefill: bool,
        kv_cache: KVCache,
        cache_position: int
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        kv_cache.update_kv_states(
            key_states=key_states, 
            value_states=value_states, 
            is_prefill=is_prefill,
            cache_position=cache_position
        )

        key_states, value_states = kv_cache.get_kv_states(cache_position=cache_position)
        key_states.to(hidden_states.device)
        value_states.to(hidden_states.device)

        key_states = repeat_kv(key_states, self.num_groups)
        value_states = repeat_kv(value_states, self.num_groups)

        query_states = query_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.scaling,
            is_causal=is_prefill,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output

class DecoderBlock(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.kv_cache = KVCache(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache_position: int
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output = self.self_attn(
            hidden_states,
            position_embeddings,
            is_prefill,
            self.kv_cache,
            cache_position
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(hidden_states)
        return residual + mlp_output

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_hidden_layers)]  
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryPositionEmbedding(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        is_prefill: bool,
        cache_position: int,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        if is_prefill:
            position_ids = torch.arange(cache_position, device=hidden_states.device).unsqueeze(0)
        else:
            position_ids = torch.tensor([[cache_position-1]], device=hidden_states.device)
        
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                is_prefill=is_prefill,
                position_embeddings=position_embeddings,
                cache_position=cache_position
            )
        
        return self.norm(hidden_states)

class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        is_prefill: bool,
        cache_position: int,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids, 
            is_prefill=is_prefill, 
            cache_position=cache_position
        )
        logits = self.lm_head(hidden_states)
        return logits


    



