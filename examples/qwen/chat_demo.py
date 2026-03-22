# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen 2.5 0.5B-Instruct chat demo on Tenstorrent Blackhole.

Runs the full model via tt-lang kernels on a single Blackhole card.
Supports interactive multi-turn chat with streaming output.

Usage:
    source build/env/activate
    python examples/qwen/chat_demo.py
    python examples/qwen/chat_demo.py --prompt "What is the meaning of life?"
"""

import argparse
import os
import sys
import time

# Suppress ttnn/tt-metal verbose logging before import
os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
os.environ.setdefault("LOGURU_LEVEL", "ERROR")

import torch

sys.path.insert(0, os.path.dirname(__file__))

import ttnn
from model import QwenModel


def get_tokenizer():
    from transformers import AutoTokenizer

    tokenizer_path = os.path.join(os.path.dirname(__file__), "weights", "tokenizer")
    if os.path.exists(tokenizer_path):
        return AutoTokenizer.from_pretrained(tokenizer_path)
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def chat_single(model, tokenizer, user_message, max_new_tokens=100):
    """Run a single chat turn."""
    messages = [{"role": "user", "content": user_message}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt)

    if len(input_ids) > model.max_seq_len - max_new_tokens:
        print(f"Warning: prompt too long ({len(input_ids)} tokens), truncating")
        input_ids = input_ids[: model.max_seq_len - max_new_tokens]

    t0 = time.time()
    generated = []
    first_token_time = None
    for token_id in model.generate(input_ids, max_new_tokens=max_new_tokens):
        if first_token_time is None:
            first_token_time = time.time()
            # Print label after prefill completes (all compilation done)
            sys.stdout.write("Assistant: ")
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
        sys.stdout.write(tokenizer.decode([token_id]))
        sys.stdout.flush()
    elapsed = time.time() - t0
    print()

    n = len(generated)
    if n > 0:
        prefill_time = (first_token_time - t0) if first_token_time else elapsed
        decode_time = elapsed - prefill_time
        decode_tps = (n - 1) / decode_time if n > 1 and decode_time > 0 else 0
        print(f"  [{n} tokens | prefill: {prefill_time:.1f}s | decode: {decode_time:.1f}s | {decode_tps:.2f} tok/s]")

    return generated


def interactive_mode(model, tokenizer, max_new_tokens=100):
    """Interactive chat loop."""
    print("=" * 60)
    print("  Qwen 2.5 0.5B-Instruct on Tenstorrent Blackhole")
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        chat_single(model, tokenizer, user_input, max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Qwen chat on Blackhole")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Max tokens to generate")
    args = parser.parse_args()

    tokenizer = get_tokenizer()

    # Check weights exist
    from setup import is_ready
    if not is_ready():
        print("Weights not found. Run setup first:\n")
        print("  python examples/qwen/setup.py\n")
        sys.exit(1)

    device = ttnn.open_device(device_id=0)
    try:
        model = QwenModel(device)

        model.quiet = True
        print(f"(Compilation output → {model._compile_log})\n")

        if args.prompt:
            chat_single(model, tokenizer, args.prompt, args.max_tokens)
        else:
            interactive_mode(model, tokenizer, args.max_tokens)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
