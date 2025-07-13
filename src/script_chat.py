"""
Interactive Script for Running Qwen3 Model
Author: Bound
Date: May 30, 2025
Version: 1.0
"""

import torch
import argparse

from generate import naive_generate
from load import model_load

def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen3 in interactive mode.")
    parser.add_argument("-m", "--max_length", type=int, default=1024, help="Maximum length of the generated output.")
    parser.add_argument("-t", "--deep_think", action='store_true', help="Enable thinking mode")
    parser.add_argument("-p", "--checkpoint", type=str, default="../weight/Qwen3-0.6B_weights.pth", help="Model checkpoint file path.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Running model on which device")
    parser.add_argument("-n", "--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Which official model to use")
    return parser.parse_args()

def main():
    args = parse_args()
    model, tokenizer = model_load(model_name=args.model_name, device=args.device, checkpoint=args.checkpoint)
    
    print(f"💡 Deep thinking mode: {'Enabled' if args.deep_think else 'Disabled'}")
    print("🔁 Enter your prompts below. Type 'exit' to quit.")
    
    while True:
        try:
            prompt = input("\nUser: ")
            if prompt.lower() == "exit":
                print("👋 Exiting...")
                break
            
            messages = [
                {"role": "user", "content": prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.deep_think 
            )

            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                output_ids = naive_generate(
                    model=model,
                    input_ids=input_ids,
                    max_generate_len=args.max_length,
                )

            output_ids = output_ids[0].tolist()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\n👋 Interrupted. Exiting...")
            break

if __name__ == "__main__":
    main()








