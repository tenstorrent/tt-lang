#!/usr/bin/env python3
"""Profile the traced decode loop to find remaining bottlenecks."""

import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from model import QwenModel, TILE
from kernels.kv_cache_update_traced import build_full_masks

WARMUP = 3
ITERS = 30


def profile_decode(model, token_id, pos):
    """Profile one decode step with detailed timing breakdown."""

    # ---- Phase 1: Host tensor preparation ----
    t0 = time.perf_counter()

    x = model.embed_weight[token_id:token_id+1]
    x_padded = torch.zeros(TILE, model.hidden_size, dtype=torch.bfloat16)
    x_padded[0] = x[0].bfloat16()

    cos_pos = torch.ones(TILE, model.head_dim, dtype=torch.bfloat16)
    sin_pos = torch.zeros(TILE, model.head_dim, dtype=torch.bfloat16)
    cos_pos[0] = model.rope_cos[pos].bfloat16()
    sin_pos[0] = model.rope_sin[pos].bfloat16()

    mask_t = torch.full((TILE, model.padded_max_seq), float("-inf"), dtype=torch.bfloat16)
    mask_t[0, :pos + 1] = 0.0

    kv_row_m, kv_irow_m, kv_col_m, kv_icol_m = build_full_masks(pos)

    t_host_prep = time.perf_counter() - t0

    # ---- Phase 2: Host-to-device copies ----
    t1 = time.perf_counter()

    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(x_padded, layout=ttnn.TILE_LAYOUT), model._tb["x_a"])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(cos_pos, layout=ttnn.TILE_LAYOUT), model._tb["cos_dev"])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(sin_pos, layout=ttnn.TILE_LAYOUT), model._tb["sin_dev"])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(mask_t, layout=ttnn.TILE_LAYOUT), model._tb["mask_dev"])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(kv_row_m, layout=ttnn.TILE_LAYOUT), model._tb["kv_row_masks"])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(kv_irow_m, layout=ttnn.TILE_LAYOUT), model._tb["kv_irow_masks"])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(kv_col_m, layout=ttnn.TILE_LAYOUT), model._tb["kv_col_masks"])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(kv_icol_m, layout=ttnn.TILE_LAYOUT), model._tb["kv_icol_masks"])

    t_h2d = time.perf_counter() - t1

    # ---- Phase 3: Trace execution (24 layers + lm_head + argmax pipeline) ----
    t2 = time.perf_counter()
    ttnn.execute_trace(model.device, model._trace_id, cq_id=0, blocking=True)
    t_trace = time.perf_counter() - t2

    # ---- Phase 4: Read argmax result (tiny readback from trace buffer) ----
    t3 = time.perf_counter()
    token = model._read_traced_argmax()
    t_argmax = time.perf_counter() - t3

    model.cache_pos = pos + 1
    t_total = time.perf_counter() - t0

    return {
        "host_prep": t_host_prep,
        "h2d_copy": t_h2d,
        "trace_exec": t_trace,
        "argmax": t_argmax,
        "total": t_total,
        "token": token,
    }


def profile_h2d_breakdown(model, pos):
    """Break down host-to-device copy time per tensor."""
    x_padded = torch.zeros(TILE, model.hidden_size, dtype=torch.bfloat16)
    cos_pos = torch.ones(TILE, model.head_dim, dtype=torch.bfloat16)
    sin_pos = torch.zeros(TILE, model.head_dim, dtype=torch.bfloat16)
    mask_t = torch.full((TILE, model.padded_max_seq), float("-inf"), dtype=torch.bfloat16)
    mask_t[0, :pos + 1] = 0.0
    kv_row_m, kv_irow_m, kv_col_m, kv_icol_m = build_full_masks(pos)

    tensors = [
        ("x_padded  [32,896]", x_padded, "x_a"),
        ("cos_pos   [32,64]",  cos_pos,  "cos_dev"),
        ("sin_pos   [32,64]",  sin_pos,  "sin_dev"),
        ("mask      [32,512]", mask_t,   "mask_dev"),
        ("kv_row_m  [32,512]", kv_row_m, "kv_row_masks"),
        ("kv_irow_m [32,512]", kv_irow_m, "kv_irow_masks"),
        ("kv_col_m  [32,512]", kv_col_m, "kv_col_masks"),
        ("kv_icol_m [32,512]", kv_icol_m, "kv_icol_masks"),
    ]

    results = []
    for name, tensor, buf_key in tensors:
        nbytes = tensor.nelement() * tensor.element_size()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(tensor, layout=ttnn.TILE_LAYOUT), model._tb[buf_key])
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times) * 1e3
        results.append((name, nbytes, avg))

    return results


def profile_host_prep(model, token_id, pos):
    """Break down host tensor preparation time."""
    N = 200

    t0 = time.perf_counter()
    for _ in range(N):
        x = model.embed_weight[token_id:token_id+1]
        x_padded = torch.zeros(TILE, model.hidden_size, dtype=torch.bfloat16)
        x_padded[0] = x[0].bfloat16()
    t_embed = (time.perf_counter() - t0) / N * 1e3

    t0 = time.perf_counter()
    for _ in range(N):
        cos_pos = torch.ones(TILE, model.head_dim, dtype=torch.bfloat16)
        sin_pos = torch.zeros(TILE, model.head_dim, dtype=torch.bfloat16)
        cos_pos[0] = model.rope_cos[pos].bfloat16()
        sin_pos[0] = model.rope_sin[pos].bfloat16()
    t_rope = (time.perf_counter() - t0) / N * 1e3

    t0 = time.perf_counter()
    for _ in range(N):
        mask_t = torch.full((TILE, model.padded_max_seq), float("-inf"), dtype=torch.bfloat16)
        mask_t[0, :pos + 1] = 0.0
    t_mask = (time.perf_counter() - t0) / N * 1e3

    t0 = time.perf_counter()
    for _ in range(N):
        kv_row_m, kv_irow_m, kv_col_m, kv_icol_m = build_full_masks(pos)
    t_kvmask = (time.perf_counter() - t0) / N * 1e3

    return [
        ("Embedding + pad", t_embed),
        ("RoPE cos/sin", t_rope),
        ("Attention mask", t_mask),
        ("KV cache masks", t_kvmask),
    ]


def main():
    from transformers import AutoTokenizer

    tokenizer_path = os.path.join(os.path.dirname(__file__), "weights", "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    messages = [{"role": "user", "content": "Explain quantum computing in detail."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt)

    device = ttnn.open_device(device_id=0)
    try:
        model = QwenModel(device)
        model.quiet = True

        # Prefill
        logits = model.prefill(input_ids)
        first_token = logits[-1].argmax().item()
        print(f"Prefill done, first token: '{tokenizer.decode([first_token])}'")

        # Set up trace
        model._init_trace_buffers()
        model._capture_decode_trace()

        # Warmup
        cur_token = first_token
        for _ in range(WARMUP):
            cur_token = model.decode_step_traced(cur_token, greedy=True)

        # ============================================================
        # Main profile: decode step breakdown
        # ============================================================
        print(f"\n{'='*70}")
        print(f"Decode step breakdown ({ITERS} iterations)")
        print(f"{'='*70}")

        timings = {k: [] for k in ["host_prep", "h2d_copy", "trace_exec", "argmax", "total"]}
        for _ in range(ITERS):
            r = profile_decode(model, cur_token, model.cache_pos)
            cur_token = r["token"]
            for k in timings:
                timings[k].append(r[k])

        print(f"\n  {'Phase':<25} {'Avg (ms)':>10} {'Min (ms)':>10} {'% Total':>10}")
        print("  " + "-" * 58)
        avg_total = sum(timings["total"]) / len(timings["total"])
        for phase in ["host_prep", "h2d_copy", "trace_exec", "argmax"]:
            vals = timings[phase]
            avg = sum(vals) / len(vals) * 1e3
            mn = min(vals) * 1e3
            pct = (avg / (avg_total * 1e3)) * 100
            print(f"  {phase:<25} {avg:>10.2f} {mn:>10.2f} {pct:>9.1f}%")
        print("  " + "-" * 58)
        print(f"  {'TOTAL':<25} {avg_total*1e3:>10.2f} {min(timings['total'])*1e3:>10.2f} {'100.0%':>10}")
        print(f"\n  Throughput: {1.0 / avg_total:.1f} tok/s")

        # ============================================================
        # H2D copy breakdown
        # ============================================================
        print(f"\n{'='*70}")
        print("H2D copy breakdown (per tensor, 20 iterations each)")
        print(f"{'='*70}")
        h2d_results = profile_h2d_breakdown(model, model.cache_pos)
        total_h2d = sum(r[2] for r in h2d_results)
        print(f"  {'Tensor':<25} {'Bytes':>10} {'Avg (ms)':>10} {'% H2D':>8}")
        print("  " + "-" * 56)
        for name, nbytes, avg_ms in h2d_results:
            pct = avg_ms / total_h2d * 100
            print(f"  {name:<25} {nbytes:>10,} {avg_ms:>10.2f} {pct:>7.1f}%")
        print("  " + "-" * 56)
        print(f"  {'TOTAL':<25} {'':>10} {total_h2d:>10.2f}")

        total_bytes = sum(r[1] for r in h2d_results)
        print(f"\n  Total bytes uploaded: {total_bytes:,} ({total_bytes/1024:.1f} KB)")

        # ============================================================
        # Host prep breakdown
        # ============================================================
        print(f"\n{'='*70}")
        print("Host tensor preparation breakdown (200 iterations each)")
        print(f"{'='*70}")
        prep_results = profile_host_prep(model, cur_token, model.cache_pos)
        total_prep = sum(r[1] for r in prep_results)
        for name, t_ms in prep_results:
            pct = t_ms / total_prep * 100
            print(f"  {name:<25} {t_ms:>8.2f}ms  ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<25} {total_prep:>8.2f}ms")

        # ============================================================
        # Summary
        # ============================================================
        avg_ms = avg_total * 1e3
        print(f"\n{'='*70}")
        print("SUMMARY: Where time goes per token")
        print(f"{'='*70}")
        phases = ["host_prep", "h2d_copy", "trace_exec", "argmax"]
        for phase in phases:
            avg = sum(timings[phase]) / len(timings[phase]) * 1e3
            pct = avg / avg_ms * 100
            bar = "#" * int(pct / 2)
            print(f"  {phase:<15} {avg:>7.2f}ms ({pct:>5.1f}%) {bar}")
        print(f"  {'TOTAL':<15} {avg_ms:>7.2f}ms  → {1.0/avg_total:.1f} tok/s")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
