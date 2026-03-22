# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Phase 4: Full Qwen 2.5 0.5B-Instruct prefill.

Runs the complete model (embedding → 24 layers → lm_head) on a test prompt
and validates against HuggingFace reference.

Usage:
    source build/env/activate
    python examples/qwen/prefill.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

import ttnn
from model import QwenModel


def test_prefill():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), "weights", "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Test prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)
    print(f"Prompt: '{prompt}'")
    print(f"Tokens: {input_ids} (len={len(input_ids)})")

    # === Our model ===
    device = ttnn.open_device(device_id=0)
    try:
        print("\n=== TT-Lang Model ===")
        model = QwenModel(device)

        t0 = time.time()
        logits = model.prefill(input_ids)
        prefill_time = time.time() - t0

        print(f"\nPrefill complete in {prefill_time:.1f}s")
        print(f"Logits shape: {logits.shape}")

        # Get predicted next token
        next_token_id = logits[-1].argmax().item()
        next_token = tokenizer.decode([next_token_id])
        print(f"Next token prediction: '{next_token}' (id={next_token_id})")

        # Top-5 predictions
        top5 = torch.topk(logits[-1], 5)
        print("Top-5 predictions:")
        for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
            tok = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{tok}' (id={idx.item()}, logit={val.item():.2f})")

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
        hf_output = hf_model(hf_input)
        hf_logits = hf_output.logits[0].float()  # [seq, vocab]

    hf_next_token_id = hf_logits[-1].argmax().item()
    hf_next_token = tokenizer.decode([hf_next_token_id])
    print(f"Next token prediction: '{hf_next_token}' (id={hf_next_token_id})")

    hf_top5 = torch.topk(hf_logits[-1], 5)
    print("Top-5 predictions:")
    for i, (val, idx) in enumerate(zip(hf_top5.values, hf_top5.indices)):
        tok = tokenizer.decode([idx.item()])
        print(f"  {i+1}. '{tok}' (id={idx.item()}, logit={val.item():.2f})")

    # === Comparison ===
    print("\n=== Comparison ===")

    # Compare logits
    # Our logits may have different absolute scale but should rank tokens similarly
    our_last = logits[-1]
    hf_last = hf_logits[-1, :logits.shape[1]]

    pcc = torch.corrcoef(
        torch.stack([our_last.flatten(), hf_last.flatten()])
    )[0, 1].item()
    print(f"Last-position logits PCC: {pcc:.6f}")

    # Token match
    token_match = (next_token_id == hf_next_token_id)
    print(f"Top-1 token match: {'YES' if token_match else 'NO'}")
    print(f"  Ours: '{next_token}' (id={next_token_id})")
    print(f"  HF:   '{hf_next_token}' (id={hf_next_token_id})")

    # Compare top-5 overlap
    our_top5_ids = set(torch.topk(logits[-1], 5).indices.tolist())
    hf_top5_ids = set(torch.topk(hf_logits[-1], 5).indices.tolist())
    overlap = our_top5_ids & hf_top5_ids
    print(f"Top-5 overlap: {len(overlap)}/5")

    if pcc > 0.90:
        print("\nPASS (PCC > 0.90)")
    else:
        print(f"\nFAIL (PCC = {pcc:.6f})")


if __name__ == "__main__":
    test_prefill()
