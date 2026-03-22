# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 5: Decode with KV cache.

Tests autoregressive generation with Qwen 2.5 0.5B-Instruct:
1. Prefill a prompt
2. Generate tokens one by one using KV cache in device DRAM
3. Compare against HuggingFace greedy generation

Usage:
    source build/env/activate
    python examples/qwen/decode.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

import ttnn
from model import QwenModel


def test_greedy_generation():
    from setup import is_ready
    if not is_ready():
        print("Weights not found. Run: python examples/qwen/setup.py")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "weights", "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)
    max_new_tokens = 20

    print(f"Prompt: '{prompt}'")
    print(f"Input tokens: {input_ids} (len={len(input_ids)})")
    print(f"Generating {max_new_tokens} tokens...\n")

    # === TT-Lang generation ===
    print("=== TT-Lang Model ===")
    device = ttnn.open_device(device_id=0)
    try:
        model = QwenModel(device)

        t0 = time.time()
        generated_tokens = []
        print(f"Output: {prompt}", end="", flush=True)
        for token_id in model.generate(input_ids, max_new_tokens=max_new_tokens):
            generated_tokens.append(token_id)
            word = tokenizer.decode([token_id])
            print(word, end="", flush=True)
            # Stop on EOS
            if token_id == tokenizer.eos_token_id:
                break
        total_time = time.time() - t0

        print(f"\n\nGenerated {len(generated_tokens)} tokens in {total_time:.1f}s")
        if len(generated_tokens) > 0:
            # Prefill time is most of the first token; decode is the rest
            print(f"  ~{total_time / len(generated_tokens):.2f}s per token (including prefill)")
    finally:
        ttnn.close_device(device)

    # === HuggingFace reference ===
    print("\n=== HuggingFace Reference ===")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16
    )
    hf_model.eval()

    with torch.no_grad():
        hf_input = torch.tensor([input_ids])
        hf_output = hf_model.generate(
            hf_input,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        hf_tokens = hf_output[0].tolist()
        hf_generated = hf_tokens[len(input_ids):]

    hf_text = tokenizer.decode(hf_generated)
    print(f"Output: {prompt}{hf_text}")
    print(f"Tokens: {hf_generated}")

    # === Comparison ===
    print("\n=== Token-by-Token Comparison ===")
    match_count = 0
    for i, (ours, hf) in enumerate(zip(generated_tokens, hf_generated)):
        our_word = tokenizer.decode([ours])
        hf_word = tokenizer.decode([hf])
        match = "✓" if ours == hf else "✗"
        if ours == hf:
            match_count += 1
        print(f"  {i:2d}: {match} ours='{our_word}' ({ours}) hf='{hf_word}' ({hf})")

    total = min(len(generated_tokens), len(hf_generated))
    print(f"\nMatch rate: {match_count}/{total} ({100*match_count/max(total,1):.0f}%)")

    if match_count == total:
        print("PASS: Perfect token match!")
    elif match_count >= total * 0.7:
        print("PASS: >70% token match (bf16 precision differences expected)")
    else:
        print(f"FAIL: Only {match_count}/{total} tokens match")


def test_chat():
    """Test with instruction-tuned chat template."""
    from transformers import AutoTokenizer

    tokenizer_path = os.path.join(os.path.dirname(__file__), "weights", "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    messages = [{"role": "user", "content": "What is 2+2?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt)

    print(f"\n{'='*60}")
    print("Chat test")
    print(f"{'='*60}")
    print(f"Prompt tokens: {len(input_ids)}")

    device = ttnn.open_device(device_id=0)
    try:
        model = QwenModel(device)

        print("Assistant: ", end="", flush=True)
        generated = []
        for token_id in model.generate(input_ids, max_new_tokens=50):
            generated.append(token_id)
            word = tokenizer.decode([token_id])
            print(word, end="", flush=True)
            if token_id == tokenizer.eos_token_id:
                break
        print(f"\n({len(generated)} tokens)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_greedy_generation()
    # Uncomment to also test chat:
    # test_chat()
