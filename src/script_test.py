"""
Interactive Script for Running Qwen3 Model Performance test
Author: Bound
Date: June 3, 2025
Version: 1.0
"""

import argparse
import torch
import torch.nn as nn
import numpy as np

from model import Qwen3Config
from generate import benchmark_generate
from load import model_load

def performance_test(
    model: nn.Module,
    config: Qwen3Config,
    prompt_length: int,
    output_len: int,
    warmup: int,
    num_tests: int
) -> dict:
    device = next(model.parameters()).device

    prompt_ids = torch.randint(
        0, config.vocab_size,
        (1, prompt_length),
        dtype=torch.int64,
        device=device
    )

    for _ in range(warmup):
        with torch.no_grad():
            _ = benchmark_generate(model, prompt_ids, min(16, output_len))

    prefill_times = []
    decode_times = []

    for i in range(num_tests):
        with torch.no_grad():
            current_prefill, current_decode = benchmark_generate(
                model=model,
                input_ids=prompt_ids,
                output_len=output_len
            )

        prefill_times.append(current_prefill)
        decode_times.append(current_decode)

        if (i + 1) % max(1, num_tests // 5) == 0:
            print(f"Complete test {i+1}/{num_tests}")
    

    avg_prefill = np.mean(prefill_times)
    avg_decode = np.mean(decode_times)
    avg_per_token = avg_decode / output_len
    throughput = output_len / avg_decode

    return {
        "prefill_avg": avg_prefill * 1000,  
        "decode_total_avg": avg_decode * 1000,  
        "decode_per_token_avg": avg_per_token * 1000,  
        "decode_throughput": throughput,  
        "device": str(device)
    }


def print_results(results):
    """Print test result"""
    print("\n" + "="*80)
    print(f"{'Model Performance Test Results':^80}")
    print("="*80)
    print(f"{'Device:':<30}{results['device']}")
    print(f"{'Average Prefill Time:':<30}{results['prefill_avg']:.2f} ms")
    print("-"*80)
    print(f"{'Total Average Decode Time:':<30}{results['decode_total_avg']:.2f} ms")
    print(f"{'Average Decode Time per Token:':<30}{results['decode_per_token_avg']:.2f} ms")
    print(f"{'Decode Throughput:':<30}{results['decode_throughput']:.2f} tokens/s")
    print("-"*80)


def parse_args():
    parser = argparse.ArgumentParser(description="Model Performance Benchmarking Script (Fixed Output Length)")
    parser.add_argument("-p", "--prompt_len", type=int, default=128, help="Input prompt length in tokens")
    parser.add_argument("-o", "--output_len", type=int, default=256, help="Number of output tokens to generate")
    parser.add_argument("-w", "--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("-t", "--num_tests", type=int, default=2, help="Number of benchmark test runs")
    parser.add_argument("-c", "--checkpoint", type=str, default="../qwen3_0.6b_weights.pth", help="Model checkpoint path")
    parser.add_argument("-d", "--device", type=torch.device, default="cpu", help="Running model on which device")
    parser.add_argument("-n", "--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Which official model to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model, tokenizer = model_load(model_name=args.model_name, device=args.device, checkpoint=args.checkpoint)

    print(f"Device: {args.device}")
    print(f"Data Type: {model.config.torch_type}")
    print(f"Input Length: {args.prompt_len} tokens")
    print(f"Output Length: {args.output_len} tokens")
    print(f"Number of Tests: {args.num_tests}")
    print(f"Warmup Rounds: {args.warmup}")

    results = performance_test(
        model=model,
        config=model.config,
        prompt_length=args.prompt_len,
        output_len=args.output_len,
        warmup=args.warmup,
        num_tests=args.num_tests
    )
    print_results(results)
