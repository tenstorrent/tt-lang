#!/usr/bin/env python3
"""Profile individual kernel execution times within the decode pipeline.

Runs kernels outside the trace with device synchronization barriers
to measure each one independently.
"""

import os
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from model import QwenModel, TILE
from kernels.argmax import (
    parallel_max_reduce_kernel, global_max_reduce_kernel,
    parallel_index_find_kernel,
)

WARMUP = 3
ITERS = 10


def profile_one_layer(model, layer_idx, tb, w):
    """Profile all kernels in a single transformer layer."""
    from kernels.linear import (
        linear_bias_kernel, linear_residual_kernel,
        fused_gate_up_silu_kernel,
        down_proj_partial_kernel, down_proj_reduce_residual_kernel,
    )
    from kernels.rope import batch_rope_kernel
    from kernels.rmsnorm import fused_rmsnorm_kernel
    from kernels.kv_cache_update_traced import kv_cache_update_traced
    from kernels.multicore_attn import (
        parallel_partial_g0, parallel_partial_g1,
        parallel_reduce_g0, parallel_reduce_g1,
    )

    device = model.device
    H = model.hidden_size
    KV = model.num_kv_heads * model.head_dim
    timings = {}

    def bench(name, fn):
        fn()
        ttnn.synchronize_device(device)
        times = []
        for _ in range(ITERS):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        timings[name] = sum(times) / len(times) * 1e3

    # 1. RMSNorm (pre-attention)
    bench("rmsnorm_pre_attn", lambda:
        fused_rmsnorm_kernel(tb["x_a"], model.mean_scaler_device,
                              w["input_layernorm_weight"], tb["normed"]))

    # 2. Fused QKV projection
    bench("fused_qkv", lambda:
        linear_bias_kernel(tb["normed"], w["qkv_weight"], w["qkv_bias"], tb["qkv_out"]))

    # 3. Fused Q+K RoPE
    bench("rope_qk", lambda:
        batch_rope_kernel(tb["qkv_out"][:, :H+KV], tb["cos_dev"], tb["sin_dev"],
                          tb["qk_rot"]))

    # 4. KV cache update
    bench("kv_cache_update", lambda:
        kv_cache_update_traced(
            tb["qk_rot"][:, H:H+KV], tb["qkv_out"][:, H+KV:H+KV+KV],
            model.kv_cache_dev[layer_idx][0]["k_t"],
            model.kv_cache_dev[layer_idx][1]["k_t"],
            model.kv_cache_dev[layer_idx][0]["v"],
            model.kv_cache_dev[layer_idx][1]["v"],
            tb["kv_row_masks"], tb["kv_irow_masks"],
            tb["kv_col_masks"], tb["kv_icol_masks"]))

    # 5. Parallel attention (group 0)
    bench("attn_partial_g0", lambda:
        parallel_partial_g0(tb["qk_rot"][:, :H], model.kv_cache_dev[layer_idx][0]["k_t"],
                            model.kv_cache_dev[layer_idx][0]["v"],
                            tb["mask_dev"], model.ones_scaler_device,
                            model.attn_scale_device,
                            tb["part_m"], tb["part_d"], tb["part_o0"], tb["part_o1"]))
    bench("attn_reduce_g0", lambda:
        parallel_reduce_g0(tb["part_m"], tb["part_d"], tb["part_o0"], tb["part_o1"],
                           tb["attn_out"]))

    # 5b. Parallel attention (group 1)
    bench("attn_partial_g1", lambda:
        parallel_partial_g1(tb["qk_rot"][:, :H], model.kv_cache_dev[layer_idx][1]["k_t"],
                            model.kv_cache_dev[layer_idx][1]["v"],
                            tb["mask_dev"], model.ones_scaler_device,
                            model.attn_scale_device,
                            tb["part_m"], tb["part_d"], tb["part_o0"], tb["part_o1"]))
    bench("attn_reduce_g1", lambda:
        parallel_reduce_g1(tb["part_m"], tb["part_d"], tb["part_o0"], tb["part_o1"],
                           tb["attn_out"]))

    # 6. O projection + residual (fused)
    bench("o_proj_residual", lambda:
        linear_residual_kernel(tb["attn_out"], w["o_proj_weight"], tb["x_a"], tb["post_attn"]))

    # 7. MLP
    bench("rmsnorm_pre_mlp", lambda:
        fused_rmsnorm_kernel(tb["post_attn"], model.mean_scaler_device,
                              w["post_attention_layernorm_weight"], tb["normed2"]))
    bench("fused_gate_up_silu", lambda:
        fused_gate_up_silu_kernel(
            tb["normed2"], w["gate_proj_weight"], w["up_proj_weight"], tb["mlp_hidden"]))
    bench("down_proj_partial", lambda:
        down_proj_partial_kernel(
            tb["mlp_hidden"], w["down_proj_weight"], tb["down_proj_partial"]))
    bench("down_reduce_residual", lambda:
        down_proj_reduce_residual_kernel(
            tb["down_proj_partial"], tb["post_attn"], tb["x_b"]))

    return timings


def profile_tail(model, tb):
    """Profile final norm + lm_head + argmax."""
    from kernels.linear import linear_kernel
    from kernels.rmsnorm import fused_rmsnorm_kernel

    device = model.device
    timings = {}

    def bench(name, fn):
        fn()
        ttnn.synchronize_device(device)
        times = []
        for _ in range(ITERS):
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(device)
            times.append(time.perf_counter() - t0)
        timings[name] = sum(times) / len(times) * 1e3

    bench("final_rmsnorm", lambda:
        fused_rmsnorm_kernel(tb["x_a"], model.mean_scaler_device,
                              model.final_norm_weight, tb["final_out"]))
    bench("lm_head", lambda:
        linear_kernel(tb["final_out"], model.lm_head_weight_device, tb["logits"]))
    bench("argmax_max_reduce", lambda:
        parallel_max_reduce_kernel(tb["logits"], tb["argmax_scaler"], tb["argmax_max_out"]))
    bench("argmax_global_max", lambda:
        global_max_reduce_kernel(tb["argmax_max_out"], tb["argmax_scaler"], tb["argmax_global_max"]))
    bench("argmax_index_find", lambda:
        parallel_index_find_kernel(tb["logits"], tb["argmax_global_max"], tb["argmax_index_out"]))

    return timings


def main():
    device = ttnn.open_device(device_id=0)
    try:
        model = QwenModel(device)
        model.quiet = True

        from transformers import AutoTokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "weights", "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        messages = [{"role": "user", "content": "What is 2+2?"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(prompt)
        logits = model.prefill(input_ids)

        model._init_trace_buffers()
        model._traced_decode_layers()
        ttnn.synchronize_device(device)

        tb = model._tb
        w = model.layer_weights[0]

        print(f"\n{'='*70}")
        print(f"Per-kernel profile (layer 0, {ITERS} iterations each)")
        print(f"{'='*70}")

        layer_timings = profile_one_layer(model, 0, tb, w)

        groups = {
            "Attention pre": ["rmsnorm_pre_attn", "fused_qkv", "rope_qk"],
            "KV cache":      ["kv_cache_update"],
            "Attention":     ["attn_partial_g0", "attn_reduce_g0", "attn_partial_g1", "attn_reduce_g1"],
            "Post-attn":     ["o_proj_residual"],
            "MLP":           ["rmsnorm_pre_mlp", "fused_gate_up_silu", "down_proj_partial", "down_reduce_residual"],
        }

        total_layer = sum(layer_timings.values())
        print(f"\n  {'Kernel':<28} {'Time (ms)':>10} {'×24 (ms)':>10} {'% Layer':>10}")
        print("  " + "-" * 61)
        for group_name, kernels in groups.items():
            group_total = sum(layer_timings.get(k, 0) for k in kernels)
            print(f"  {group_name + ':':<28}")
            for k in kernels:
                t = layer_timings.get(k, 0)
                print(f"    {k:<26} {t:>10.3f} {t*24:>10.1f} {t/total_layer*100:>9.1f}%")
            print(f"    {'subtotal':<26} {group_total:>10.3f} {group_total*24:>10.1f} {group_total/total_layer*100:>9.1f}%")
            print()

        print(f"  {'LAYER TOTAL':<28} {total_layer:>10.3f} {total_layer*24:>10.1f}")
        print(f"  Kernels per layer: {len(layer_timings)}")

        print(f"\n{'='*70}")
        print(f"Tail kernels (final norm + lm_head + argmax)")
        print(f"{'='*70}")

        tail_timings = profile_tail(model, tb)
        total_tail = sum(tail_timings.values())
        print(f"\n  {'Kernel':<28} {'Time (ms)':>10}")
        print("  " + "-" * 39)
        for k, t in tail_timings.items():
            print(f"  {k:<28} {t:>10.3f}")
        print(f"  {'TAIL TOTAL':<28} {total_tail:>10.3f}")

        print(f"\n{'='*70}")
        print("FULL PIPELINE ESTIMATE")
        print(f"{'='*70}")
        est_total = total_layer * 24 + total_tail
        print(f"  24 layers:  {total_layer*24:>8.1f}ms")
        print(f"  Tail:       {total_tail:>8.1f}ms")
        print(f"  TOTAL:      {est_total:>8.1f}ms  →  {1000/est_total:.1f} tok/s (without trace)")

        all_kernels = {}
        for k, v in layer_timings.items():
            all_kernels[k + " (×24)"] = v * 24
        for k, v in tail_timings.items():
            all_kernels[k] = v

        print(f"\n  Top 5 hottest:")
        for i, (k, v) in enumerate(sorted(all_kernels.items(), key=lambda x: -x[1])[:5]):
            print(f"    {i+1}. {k:<35} {v:>8.1f}ms ({v/est_total*100:.1f}%)")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
