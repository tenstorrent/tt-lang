# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sweep using the config heuristics from bench_matmul_sweep.py.

Same shapes and timing convention as sweep.py, but replaces plan_matmul
with the baseline's choose_ksplit_plan / choose_block_cfg. For each shape
we try BOTH the baseline's ksplit plan and its SUMMA plan (Kp=1) and
report the faster of the two, matching how the baseline bench picks a
winner.

Writes /tmp/ksplit_sweep_baseline.csv and a PNG next to it.
"""

import csv
import time
from pathlib import Path

import torch
import ttnn

from ksplit_kernel import make_kernel as make_ksplit_kernel
from plot import save_plot
from summa_kernel import make_kernel as make_summa_kernel
from sweep import (
    SHAPES,
    WARMUP_RUNS,
    TIMED_RUNS,
    SLEEP_BETWEEN_MS,
    L1_BUDGET_REDUCTION_BYTES,
    FP32_ACC,
    TTNN_CFG,
    FIELDS,
    to_dev,
    pad_2d,
    time_runs,
)

TILE = 32
MAX_GRID_M = 10
MAX_GRID_N = 13

SUMMA_BLOCK_CANDIDATES = (
    (8, 8, 8),
    (8, 4, 8),
    (4, 8, 8),
    (4, 4, 8),
)
KSPLIT_BLOCK_CANDIDATES = (
    (8, 8, 8),
    (8, 8, 4),
    (8, 4, 8),
    (8, 4, 4),
    (4, 4, 8),
    (4, 4, 4),
)


def _pad_split(n_blocks, max_grid):
    cores = min(max_grid, n_blocks)
    bpn = -(-n_blocks // cores)
    return bpn, cores


def _block_shape(block_cfg):
    bm, bn, bk = block_cfg
    return bm * TILE, bn * TILE, bk * TILE


def choose_summa_block(M, K, N):
    best = None
    for block_cfg in SUMMA_BLOCK_CANDIDATES:
        bm_dim, bn_dim, bk_dim = _block_shape(block_cfg)
        if K % bk_dim or M % bm_dim or N % bn_dim:
            continue
        M_blocks = M // bm_dim
        N_blocks = N // bn_dim
        m_bpn, rows = _pad_split(M_blocks, MAX_GRID_M)
        n_bpn, cols = _pad_split(N_blocks, MAX_GRID_N)
        padded = rows * m_bpn * cols * n_bpn
        real = M_blocks * N_blocks
        score = (rows * cols) / (padded / real)
        bm, bn, bk = block_cfg
        cand = (score, bm * bn * bk, bm * bn, bk, block_cfg)
        if best is None or cand > best:
            best = cand
    if best is None:
        raise ValueError(f"No SUMMA block for M={M} K={K} N={N}")
    return best[-1]


def choose_ksplit(M, K, N):
    if M == 8192 and K == 8192 and N == 8192:
        return (8, 8, 8), (8, 6, 2)
    best = None
    for block_cfg in KSPLIT_BLOCK_CANDIDATES:
        bm_dim, bn_dim, bk_dim = _block_shape(block_cfg)
        if M % bm_dim or N % bn_dim or K % bk_dim:
            continue
        M_b, N_b, K_b = M // bm_dim, N // bn_dim, K // bk_dim
        for m_parts in range(1, min(MAX_GRID_M, M_b) + 1):
            m_span = -(-M_b // m_parts)
            for n_parts in range(1, min(MAX_GRID_N, N_b) + 1):
                max_k = MAX_GRID_N // n_parts
                if max_k == 0:
                    continue
                n_span = -(-N_b // n_parts)
                pad = (m_parts * m_span * n_parts * n_span) / (M_b * N_b)
                if M >= 4096 and N >= 4096:
                    if max_k < 2 or K_b % 2:
                        continue
                    k_range = (2,)
                else:
                    k_range = range(max_k, 0, -1)
                for k_parts in k_range:
                    if K_b % k_parts:
                        continue
                    cores = m_parts * n_parts * k_parts
                    cand = (
                        cores / pad,
                        -k_parts,
                        -pad,
                        block_cfg[0] * block_cfg[1],
                        block_cfg[2],
                        cores,
                        block_cfg[0] * block_cfg[1] * block_cfg[2],
                        block_cfg,
                        (m_parts, n_parts, k_parts),
                    )
                    if best is None or cand > best:
                        best = cand
    if best is None:
        raise ValueError(f"No ksplit plan for M={M} K={K} N={N}")
    return best[-2], best[-1]


def summa_padded_dims(M, N, block_cfg):
    bm_dim, bn_dim, _ = _block_shape(block_cfg)
    M_b, N_b = M // bm_dim, N // bn_dim
    m_bpn, rows = _pad_split(M_b, MAX_GRID_M)
    n_bpn, cols = _pad_split(N_b, MAX_GRID_N)
    return rows * m_bpn * bm_dim, cols * n_bpn * bn_dim, (rows, cols), m_bpn, n_bpn


def ksplit_padded_dims(M, N, block_cfg, part_cfg):
    bm_dim, bn_dim, _ = _block_shape(block_cfg)
    M_b, N_b = M // bm_dim, N // bn_dim
    Mp, Np, _ = part_cfg
    M_BPN = -(-M_b // Mp)
    N_BPN = -(-N_b // Np)
    return Mp * M_BPN * bm_dim, Np * N_BPN * bn_dim


def bench_variant(device, M, K, N, a_t, w_t, block_cfg, part_cfg):
    M_pad, N_pad = ksplit_padded_dims(M, N, block_cfg, part_cfg)
    Kp = part_cfg[2]
    a = to_dev(pad_2d(a_t, M_pad, K), device)
    w = to_dev(pad_2d(w_t, K, N_pad), device)
    o = to_dev(torch.zeros(M_pad, N_pad, dtype=torch.bfloat16), device)
    make = make_summa_kernel if Kp == 1 else make_ksplit_kernel
    fn = make(M_pad, K, N_pad, block_cfg, part_cfg)
    t = time_runs(thunk=lambda: fn(a, w, o), cleanup=lambda _r: None, device=device)
    for x in (a, w, o):
        ttnn.deallocate(x)
    return t


def bench_shape(device, label, M, K, N):
    torch.manual_seed(0)
    a_t = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    w_t = torch.randn(K, N, dtype=torch.bfloat16) * 0.02

    summa_block = choose_summa_block(M, K, N)
    Mp_s, Np_s = (
        _pad_split(M // (summa_block[0] * TILE), MAX_GRID_M)[1],
        _pad_split(N // (summa_block[1] * TILE), MAX_GRID_N)[1],
    )
    summa_part = (Mp_s, Np_s, 1)
    summa_t = bench_variant(device, M, K, N, a_t, w_t, summa_block, summa_part)

    ks_block, ks_part = choose_ksplit(M, K, N)
    ks_t = bench_variant(device, M, K, N, a_t, w_t, ks_block, ks_part)

    if summa_t <= ks_t:
        block_cfg, part_cfg, best_t = summa_block, summa_part, summa_t
        pick = "summa"
    else:
        block_cfg, part_cfg, best_t = ks_block, ks_part, ks_t
        pick = "ksplit"

    a_ref = to_dev(a_t, device)
    w_ref = to_dev(w_t, device)
    ttnn_t = time_runs(
        thunk=lambda: ttnn.matmul(a_ref, w_ref, compute_kernel_config=TTNN_CFG),
        cleanup=ttnn.deallocate,
        device=device,
    )
    for x in (a_ref, w_ref):
        ttnn.deallocate(x)

    bm, bn, bk = block_cfg
    Mp, Np, Kp = part_cfg
    return {
        "label": label,
        "M": M,
        "K": K,
        "N": N,
        "bm": bm,
        "bn": bn,
        "bk": bk,
        "Mp": Mp,
        "Np": Np,
        "Kp": Kp,
        "cores": Mp * Np * Kp,
        "iter_per_core": -(-(M // (bm * TILE)) // Mp) * -(-(N // (bn * TILE)) // Np),
        "pad": round(summa_t if pick == "summa" else ks_t, 4),
        "ksplit_ms": round(best_t * 1000, 4),
        "ttnn_ms": round(ttnn_t * 1000, 4),
        "ratio": round(best_t / ttnn_t, 4),
        "pcc": round(float("nan"), 6),
        "summa_ms": round(summa_t * 1000, 4),
        "ksplit_alt_ms": round(ks_t * 1000, 4),
        "pick": pick,
    }


OUTPUT_CSV = Path("/tmp/ksplit_sweep_baseline.csv")

EXTRA_FIELDS = FIELDS + ("summa_ms", "ksplit_alt_ms", "pick")


def main():
    default_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    device = ttnn.open_device(
        device_id=0,
        worker_l1_size=default_l1 - L1_BUDGET_REDUCTION_BYTES,
    )
    results = []
    try:
        for M, K, N, label in SHAPES:
            try:
                r = bench_shape(device, label, M, K, N)
            except Exception as e:
                print(f"{label:<32}  FAIL: {e}", flush=True)
                continue
            print(
                f"{label:<32}  "
                f"best={r['ksplit_ms']:>8.3f}ms ({r['pick']})  "
                f"ttnn={r['ttnn_ms']:>8.3f}ms  ratio={r['ratio']:.3f}  "
                f"summa={r['summa_ms']:.3f} ksplit={r['ksplit_alt_ms']:.3f}  "
                f"({r['bm']},{r['bn']},{r['bk']})/"
                f"({r['Mp']},{r['Np']},{r['Kp']}) cores={r['cores']}",
                flush=True,
            )
            results.append(r)
    finally:
        ttnn.close_device(device)

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote {len(results)} rows to {OUTPUT_CSV}", flush=True)

    save_plot(results, path=str(OUTPUT_CSV.with_suffix(".png")))


if __name__ == "__main__":
    main()
