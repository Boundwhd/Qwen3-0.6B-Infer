"""
Qwen3 Model Generation Module
Author: Bound
Date: May 16, 2025
Version: 1.0
"""

import torch
import torch.nn as nn
import time
from typing import Tuple

def naive_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_generate_len: int
) -> torch.Tensor:

    current_cache_position = input_ids.size(1)
    output_ids = input_ids.clone()
    
    logits = model(
        input_ids=input_ids, 
        is_prefill=True, 
        cache_position=current_cache_position
    )

    for _ in range(max_generate_len):
        next_token_logits = logits[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
        
        if torch.all(next_tokens == model.config.eos_token_id):
            break
            
        output_ids = torch.cat([output_ids, next_tokens], dim=-1)
        current_cache_position += 1
        
        logits = model(
            input_ids=next_tokens, 
            is_prefill=False, 
            cache_position=current_cache_position
        )

    return output_ids[:,input_ids.size(1):]



def benchmark_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    output_len: int
) -> Tuple[float, float]:
    
    current_cache_position = input_ids.size(1)

    next_token = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, 1),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )

    prefill_start = time.perf_counter()
    logits = model(
        input_ids=input_ids, 
        is_prefill=True, 
        cache_position=current_cache_position
    )
    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start

    decode_time = 0.0
    for _ in range(output_len):
        current_cache_position += 1
        decode_start = time.perf_counter()
        logits = model(
            input_ids=next_token, 
            is_prefill=False, 
            cache_position=current_cache_position
        )
        decode_end = time.perf_counter()
        decode_time += (decode_end - decode_start)

        next_token = logits[:, -1:, :].argmax(dim=-1).detach()
    
    del logits, next_token
    return prefill_time, decode_time