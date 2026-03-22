#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profile one decode token: measure wall time per kernel call category."""

import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from model import QwenModel

TILE = 32


def profile_decode_token(model, pos, token_id):
    """Run one decode token with detailed timing."""
    import math
    from kernels.linear import linear_kernel, linear_bias_kernel
    from kernels.elementwise import add_kernel, silu_mul_kernel
    from kernels.rope import batch_rope_kernel
    from kernels.rmsnorm import fused_device_rmsnorm
    from kernels.group_attn import head_attn_kernels
    from kernels.kv_cache_update import get_kv_cache_update_kernel

    timings = {
        "embedding": 0,
        "rope_setup": 0,
        "mask_setup": 0,
        "rmsnorm": 0,
        "qkv_proj": 0,
        "rope": 0,
        "kv_cache_update": 0,
        "attention": 0,
        "o_proj": 0,
        "residual_add": 0,
        "mlp": 0,
        "final_norm": 0,
        "lm_head": 0,
    }

    heads_per_group = model.num_q_heads // model.num_kv_heads

    # Embedding
    t0 = time.perf_counter()
    x = model.embed_weight[token_id:token_id+1]
    x_padded = torch.zeros(TILE, model.hidden_size, dtype=torch.bfloat16)
    x_padded[0] = x[0].bfloat16()
    timings["embedding"] += time.perf_counter() - t0

    # RoPE setup
    t0 = time.perf_counter()
    cos_pos = torch.ones(TILE, model.head_dim, dtype=torch.bfloat16)
    sin_pos = torch.zeros(TILE, model.head_dim, dtype=torch.bfloat16)
    cos_pos[0] = model.rope_cos[pos].bfloat16()
    sin_pos[0] = model.rope_sin[pos].bfloat16()
    timings["rope_setup"] += time.perf_counter() - t0

    with model._suppress_output():
        t0 = time.perf_counter()
        x_device = model._to_device(x_padded)
        model._decode_cos_dev = model._to_device(cos_pos)
        model._decode_sin_dev = model._to_device(sin_pos)
        timings["embedding"] += time.perf_counter() - t0

        # Mask setup (once per token)
        t0 = time.perf_counter()
        cache_len = model.padded_max_seq
        decode_mask = torch.full((TILE, cache_len), float("-inf"), dtype=torch.bfloat16)
        decode_mask[0, :pos + 1] = 0.0
        model._decode_mask_dev = model._to_device(decode_mask)
        timings["mask_setup"] += time.perf_counter() - t0

        # Per layer
        for layer_idx in range(model.num_layers):
            w = model.layer_weights[layer_idx]

            # RMSNorm
            t0 = time.perf_counter()
            normed = fused_device_rmsnorm(x_device, w["input_layernorm_weight"],
                                           model.mean_scaler_device, model.device)
            ttnn.synchronize_device(model.device)
            timings["rmsnorm"] += time.perf_counter() - t0

            # QKV projections
            t0 = time.perf_counter()
            q_out = model._alloc_zeros((TILE, model.hidden_size))
            linear_bias_kernel(normed, w["q_proj_weight"], w["q_proj_bias"], q_out)
            k_out = model._alloc_zeros((TILE, model.num_kv_heads * model.head_dim))
            linear_bias_kernel(normed, w["k_proj_weight"], w["k_proj_bias"], k_out)
            v_out = model._alloc_zeros((TILE, model.num_kv_heads * model.head_dim))
            linear_bias_kernel(normed, w["v_proj_weight"], w["v_proj_bias"], v_out)
            ttnn.synchronize_device(model.device)
            timings["qkv_proj"] += time.perf_counter() - t0

            # Batch RoPE
            t0 = time.perf_counter()
            q_rot = model._alloc_zeros((TILE, model.hidden_size))
            batch_rope_kernel(q_out, model._decode_cos_dev, model._decode_sin_dev, q_rot)
            k_rot = model._alloc_zeros((TILE, model.num_kv_heads * model.head_dim))
            batch_rope_kernel(k_out, model._decode_cos_dev, model._decode_sin_dev, k_rot)
            ttnn.synchronize_device(model.device)
            timings["rope"] += time.perf_counter() - t0

            # KV cache update
            t0 = time.perf_counter()
            tile_slot = pos // TILE
            sub_pos = pos % TILE
            update_kernel = get_kv_cache_update_kernel(tile_slot)
            update_kernel(
                k_rot, v_out,
                model.kv_cache_dev[layer_idx][0]["k_t"],
                model.kv_cache_dev[layer_idx][1]["k_t"],
                model.kv_cache_dev[layer_idx][0]["v"],
                model.kv_cache_dev[layer_idx][1]["v"],
                model._row_masks[sub_pos], model._inv_row_masks[sub_pos],
                model._col_masks[sub_pos], model._inv_col_masks[sub_pos],
            )
            ttnn.synchronize_device(model.device)
            timings["kv_cache_update"] += time.perf_counter() - t0

            # Attention (14 heads)
            t0 = time.perf_counter()
            decode_mask_dev = model._decode_mask_dev
            attn_out_device = model._alloc_zeros((TILE, model.hidden_size))
            for kv_idx in range(model.num_kv_heads):
                k_t_dev = model.kv_cache_dev[layer_idx][kv_idx]["k_t"]
                v_dev_cache = model.kv_cache_dev[layer_idx][kv_idx]["v"]
                for q_local in range(heads_per_group):
                    q_idx = kv_idx * heads_per_group + q_local
                    head_attn_kernels[q_idx](
                        q_rot, k_t_dev, v_dev_cache,
                        decode_mask_dev, model.ones_scaler_device,
                        model.attn_scale_device, attn_out_device,
                    )
            ttnn.synchronize_device(model.device)
            timings["attention"] += time.perf_counter() - t0

            # O projection
            t0 = time.perf_counter()
            proj_out = model._alloc_zeros((TILE, model.hidden_size))
            linear_kernel(attn_out_device, w["o_proj_weight"], proj_out)
            ttnn.synchronize_device(model.device)
            timings["o_proj"] += time.perf_counter() - t0

            # Residual add
            t0 = time.perf_counter()
            post_attn = model._alloc_zeros((TILE, model.hidden_size))
            add_kernel(x_device, proj_out, post_attn)
            ttnn.synchronize_device(model.device)
            timings["residual_add"] += time.perf_counter() - t0

            # MLP (rmsnorm + gate + up + silu_mul + down + add)
            t0 = time.perf_counter()
            x_device = model._run_mlp(post_attn, layer_idx, TILE, decode=True)
            ttnn.synchronize_device(model.device)
            timings["mlp"] += time.perf_counter() - t0

        # Final norm
        t0 = time.perf_counter()
        x_device = fused_device_rmsnorm(x_device, model.final_norm_weight,
                                         model.mean_scaler_device, model.device)
        x_host = ttnn.to_torch(x_device).float()
        timings["final_norm"] += time.perf_counter() - t0

    # lm_head
    t0 = time.perf_counter()
    embed_w = model.embed_weight[:model.vocab_size]
    logits = x_host[:1] @ embed_w.t()
    timings["lm_head"] += time.perf_counter() - t0

    model.cache_pos = pos + 1
    return logits, timings


def main():
    device = ttnn.open_device(device_id=0)
    model = QwenModel(device)
    model.quiet = True

    # Prefill
    input_ids = [785, 6722, 315, 9625, 374]  # "The capital of France is"
    with model._suppress_output():
        logits = model.prefill(input_ids)
    first_token = logits[-1].argmax().item()
    print(f"Prefill done. First token: {first_token}")

    # Warm up decode (compile kernels)
    print("Warming up (compiling kernels)...")
    _, _ = profile_decode_token(model, model.cache_pos, first_token)

    # Profile 5 decode tokens
    print("\nProfiling 5 decode tokens...\n")
    total_timings = None
    pos = model.cache_pos
    token_id = logits[-1].argmax().item()

    for i in range(5):
        logits, timings = profile_decode_token(model, pos, token_id)
        token_id = logits[0].argmax().item()
        pos += 1

        if total_timings is None:
            total_timings = {k: 0.0 for k in timings}
        for k, v in timings.items():
            total_timings[k] += v

    # Report
    total = sum(total_timings.values())
    per_token = total / 5
    print(f"{'Category':<20} {'Total (5 tok)':>12} {'Per token':>12} {'%':>8}")
    print("=" * 56)
    for k, v in sorted(total_timings.items(), key=lambda x: -x[1]):
        print(f"{k:<20} {v*1000:>10.1f}ms {v/5*1000:>10.1f}ms {v/total*100:>7.1f}%")
    print("=" * 56)
    print(f"{'TOTAL':<20} {total*1000:>10.1f}ms {per_token*1000:>10.1f}ms {100.0:>7.1f}%")
    print(f"\nThroughput: {5/total:.2f} tok/s")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
