# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul performance benchmark: baseline vs compiler-generated K loop.

Compares identical problem sizes with two strategies:
  1. manual_k: User-written K loop with prev + a @ b (DST accumulation,
     per-tile, no subblocking for accumulating computes).
  2. compiler_k: o.store(a @ b) with K_block in DFB. Compiler generates
     the K reduction loop. After the L1 acc implementation, this will use
     hybrid subblocking (DST within subblock, L1 acc across K).

Results are appended to _examples/matmul_perf_results.csv for tracking
across commits.
"""

import csv
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import ttl
import ttnn
from utils.correctness import assert_pcc

TILE = 32
PROFILER_CSV = Path(
    "third-party/tt-metal/generated/profiler/.logs/profile_log_device.csv"
)
RESULTS_CSV = Path("examples/matmul_bench/matmul_perf_results.csv")

PCC_THRESHOLD = 0.999  # Correlation threshold for correctness assertion


def parse_kernel_cycles(csv_path):
    """Parse profiler CSV: max kernel duration per RISC thread across all cores."""
    thread_max = defaultdict(int)
    zone_starts = {}

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # ARCH header
        next(reader)  # column header
        for row in reader:
            if len(row) < 13:
                continue
            try:
                risc = row[3]
                timestamp = int(row[5])
                zone = row[10]
                ztype = row[11]
            except (ValueError, IndexError):
                continue
            if not zone or "KERNEL" not in zone:
                continue
            core_x, core_y = row[1], row[2]
            key = (core_x, core_y, risc, zone)
            if ztype == "ZONE_START":
                zone_starts[key] = timestamp
            elif ztype == "ZONE_END" and key in zone_starts:
                dur = timestamp - zone_starts[key]
                thread_max[risc] = max(thread_max[risc], dur)
                del zone_starts[key]

    return dict(thread_max)


def get_git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def run_benchmark(name, kernel_fn, args, device, config, warmup=3, iters=10):
    """Run kernel, report timing, append to CSV.

    Warmup runs trigger compilation (first call) and stabilize caches.
    Timed iterations exclude compilation entirely. Reports median, min,
    and average to identify noise.
    """
    # Warmup: first call compiles; subsequent calls warm device caches.
    for _ in range(warmup):
        kernel_fn(*args)
        ttnn.synchronize_device(device)

    # Timed iterations.
    times = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        kernel_fn(*args)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_sorted = sorted(times)
    avg_ms = 1000 * sum(times) / len(times)
    min_ms = 1000 * times_sorted[0]
    max_ms = 1000 * times_sorted[-1]
    median_ms = 1000 * times_sorted[len(times) // 2]

    thread_cycles = {}
    if PROFILER_CSV.exists():
        thread_cycles = parse_kernel_cycles(PROFILER_CSV)

    dm_cycles = max((thread_cycles.get("BRISC", 0), thread_cycles.get("NCRISC", 0)))
    compute_cycles = max(
        thread_cycles.get(t, 0) for t in ("TRISC_0", "TRISC_1", "TRISC_2")
    )

    print(f"\n  {name}")
    print(
        f"    Host: {median_ms:.2f} ms median, {min_ms:.2f} ms min, "
        f"{max_ms:.2f} ms max, {avg_ms:.2f} ms avg "
        f"({iters} runs, {warmup} warmup)"
    )
    if thread_cycles:
        print(f"    Device: DM={dm_cycles:,}  Compute={compute_cycles:,} cycles")
        if dm_cycles > 0 and compute_cycles > 0:
            if dm_cycles > compute_cycles:
                print(
                    f"    {100*(dm_cycles-compute_cycles)/dm_cycles:.0f}% memory bound"
                )
            else:
                print(
                    f"    {100*(compute_cycles-dm_cycles)/compute_cycles:.0f}% compute bound"
                )

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "git_sha",
                    "name",
                    "strategy",
                    "M",
                    "K",
                    "N",
                    "M_block",
                    "K_block",
                    "N_block",
                    "host_median_ms",
                    "host_min_ms",
                    "host_max_ms",
                    "host_avg_ms",
                    "warmup",
                    "iters",
                    "dm_max_cycles",
                    "compute_max_cycles",
                    "NCRISC_cycles",
                    "BRISC_cycles",
                    "TRISC_0_cycles",
                    "TRISC_1_cycles",
                    "TRISC_2_cycles",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                get_git_sha(),
                name,
                config.get("strategy", ""),
                config.get("M", ""),
                config.get("K", ""),
                config.get("N", ""),
                config.get("M_block", ""),
                config.get("K_block", ""),
                config.get("N_block", ""),
                f"{median_ms:.2f}",
                f"{min_ms:.2f}",
                f"{max_ms:.2f}",
                f"{avg_ms:.2f}",
                warmup,
                iters,
                dm_cycles,
                compute_cycles,
                thread_cycles.get("NCRISC", ""),
                thread_cycles.get("BRISC", ""),
                thread_cycles.get("TRISC_0", ""),
                thread_cycles.get("TRISC_1", ""),
                thread_cycles.get("TRISC_2", ""),
            ]
        )

    return median_ms, thread_cycles


def run_ttnn_matmul_benchmark(name, a, b, device, config, warmup=3, iters=10):
    """Benchmark ttnn.matmul as reference baseline."""
    for _ in range(warmup):
        out = ttnn.matmul(a, b)
        ttnn.synchronize_device(device)
        out.deallocate()

    times = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = ttnn.matmul(a, b)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        out.deallocate()

    times_sorted = sorted(times)
    avg_ms = 1000 * sum(times) / len(times)
    min_ms = 1000 * times_sorted[0]
    max_ms = 1000 * times_sorted[-1]
    median_ms = 1000 * times_sorted[len(times) // 2]

    print(f"\n  {name}")
    print(
        f"    Host: {median_ms:.2f} ms median, {min_ms:.2f} ms min, "
        f"{max_ms:.2f} ms max, {avg_ms:.2f} ms avg "
        f"({iters} runs, {warmup} warmup)"
    )

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp",
                    "git_sha",
                    "name",
                    "strategy",
                    "M",
                    "K",
                    "N",
                    "M_block",
                    "K_block",
                    "N_block",
                    "host_median_ms",
                    "host_min_ms",
                    "host_max_ms",
                    "host_avg_ms",
                    "warmup",
                    "iters",
                    "dm_max_cycles",
                    "compute_max_cycles",
                    "NCRISC_cycles",
                    "BRISC_cycles",
                    "TRISC_0_cycles",
                    "TRISC_1_cycles",
                    "TRISC_2_cycles",
                ]
            )
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                get_git_sha(),
                name,
                "ttnn_matmul",
                config.get("M", ""),
                config.get("K", ""),
                config.get("N", ""),
                "",
                "",
                "",
                f"{median_ms:.2f}",
                f"{min_ms:.2f}",
                f"{max_ms:.2f}",
                f"{avg_ms:.2f}",
                warmup,
                iters,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )

    return median_ms


# ---- Kernel factories ----

from minimal_matmul import (
    make_matmul_v1 as make_v1,
    make_matmul_v2 as make_v2,
    make_matmul_v3 as make_v3,
    make_matmul_v4 as make_v4,
)

K_BLOCKS = [8, 4, 2, 1]


def sweep_k_blocks(
    variant_name,
    make_fn,
    Kt,
    M_block,
    N_block,
    a,
    b,
    golden,
    device,
    cfg,
    strategy,
    fp32_acc=None,
):
    """Run a kernel variant across multiple K_block values. Returns {K_block: median_ms}."""
    results = {}
    for kb in K_BLOCKS:
        if Kt % kb != 0:
            continue
        k_num = Kt // kb
        label = f"{variant_name} K={kb} Kblocks={k_num}"
        out = ttnn.from_torch(
            torch.zeros(cfg["M"], cfg["N"], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=a.memory_config(),
        )
        if fp32_acc is not None:
            kernel = make_fn(M_block, kb, N_block, fp32_acc=fp32_acc)
        else:
            kernel = make_fn(M_block, kb, N_block)
        median, _ = run_benchmark(
            label,
            kernel,
            (a, b, out),
            device,
            config={**cfg, "K_block": kb, "strategy": strategy},
        )
        assert_pcc(golden, ttnn.to_torch(out).float(), threshold=PCC_THRESHOLD)
        results[kb] = median
    return results


def main():
    device = ttnn.open_device(device_id=0)
    try:
        print("=" * 70)
        print("MATMUL PERFORMANCE BENCHMARK")
        print("=" * 70)
        print(f"  Git: {get_git_sha()}")
        print(f"  Results: {RESULTS_CSV}")

        # ---------------------------------------------------------------
        # DRAM: End-to-End Performance (Section 3.2)
        # Compare single-reader vs split-dma, K_block=8 vs K_block=1.
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("DRAM: End-to-End Performance")
        print("  4096x4096x4096 (128x128x128 tiles), blocks 8x8")
        print("=" * 70)

        Mt, Kt, Nt = 128, 128, 128
        M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
        a_torch = torch.randn(M, K, dtype=torch.bfloat16)
        b_torch = torch.randn(K, N, dtype=torch.bfloat16)
        golden = (a_torch.float() @ b_torch.float()).float()

        a = ttnn.from_torch(
            a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        b = ttnn.from_torch(
            b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        cfg = {"M": M, "K": K, "N": N, "M_block": 8, "N_block": 8}

        ttnn_t3 = run_ttnn_matmul_benchmark(
            "ttnn.matmul (reference)",
            a,
            b,
            device,
            config=cfg,
        )

        v1_results = sweep_k_blocks(
            "v1_baseline",
            make_v1,
            Kt,
            8,
            8,
            a,
            b,
            golden,
            device,
            cfg,
            strategy="single_reader",
        )
        v2_results = sweep_k_blocks(
            "v2_split_dma",
            make_v2,
            Kt,
            8,
            8,
            a,
            b,
            golden,
            device,
            cfg,
            strategy="manual_k_loop",
        )
        v4_results = sweep_k_blocks(
            "v4_l1_acc",
            make_v4,
            Kt,
            8,
            8,
            a,
            b,
            golden,
            device,
            cfg,
            strategy="l1_acc",
        )

        print(f"\n  Ratios (tt-lang / ttnn.matmul):")
        for kb in K_BLOCKS:
            if kb in v1_results:
                print(f"    v1_baseline  K={kb}: {v1_results[kb]/ttnn_t3:.2f}x")
        for kb in K_BLOCKS:
            if kb in v2_results:
                print(f"    v2_split_dma K={kb}: {v2_results[kb]/ttnn_t3:.2f}x")
        for kb in K_BLOCKS:
            if kb in v4_results:
                print(f"    v4_l1_acc    K={kb}: {v4_results[kb]/ttnn_t3:.2f}x")

        # ---------------------------------------------------------------
        # DRAM 2048^3 (same size as L1-only, for direct comparison)
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("DRAM: 2048x2048x2048 (64x64x64 tiles), blocks 8x8")
        print("  Same size as L1-only test for direct comparison.")
        print("=" * 70)

        Mt, Kt, Nt = 64, 64, 64
        M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
        a_torch = torch.randn(M, K, dtype=torch.bfloat16)
        b_torch = torch.randn(K, N, dtype=torch.bfloat16)
        golden = (a_torch.float() @ b_torch.float()).float()

        a = ttnn.from_torch(
            a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        b = ttnn.from_torch(
            b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        cfg = {"M": M, "K": K, "N": N, "M_block": 8, "N_block": 8}

        ttnn_t_dram2k = run_ttnn_matmul_benchmark(
            "ttnn.matmul (reference)",
            a,
            b,
            device,
            config=cfg,
        )

        v4_dram2k = sweep_k_blocks(
            "v4_l1_acc",
            make_v4,
            Kt,
            8,
            8,
            a,
            b,
            golden,
            device,
            cfg,
            strategy="l1_acc_dram_2k",
        )

        print(f"\n  Ratios (tt-lang / ttnn.matmul), DRAM 2048^3:")
        for kb in K_BLOCKS:
            if kb in v4_dram2k:
                print(f"    v4_l1_acc K={kb}: {v4_dram2k[kb]/ttnn_t_dram2k:.2f}x")

        # ---------------------------------------------------------------
        # L1-Only: Compute Isolation (Section 3.1)
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("L1-Only: Compute Isolation")
        print("  2048x2048x2048 (64x64x64 tiles), blocks 8x8")
        print("=" * 70)

        # Reuse a_torch, b_torch, golden from DRAM 2048^3 above.
        Mt, Kt, Nt = 64, 64, 64
        M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
        b_torch = torch.randn(K, N, dtype=torch.bfloat16)
        golden = (a_torch.float() @ b_torch.float()).float()

        l1_cfg = ttnn.L1_MEMORY_CONFIG
        a_l1 = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=l1_cfg,
        )
        b_l1 = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=l1_cfg,
        )

        cfg = {"M": M, "K": K, "N": N, "M_block": 8, "N_block": 8}

        ttnn_t4 = run_ttnn_matmul_benchmark(
            "ttnn.matmul L1 (reference)",
            a_l1,
            b_l1,
            device,
            config=cfg,
        )

        v2_l1 = sweep_k_blocks(
            "v2_split_dma",
            make_v2,
            Kt,
            8,
            8,
            a_l1,
            b_l1,
            golden,
            device,
            cfg,
            strategy="manual_k_loop_l1",
        )
        v4_l1 = sweep_k_blocks(
            "v4_l1_acc",
            make_v4,
            Kt,
            8,
            8,
            a_l1,
            b_l1,
            golden,
            device,
            cfg,
            strategy="l1_acc_l1",
        )

        print(f"\n  Ratios (tt-lang / ttnn.matmul), L1-only:")
        for kb in K_BLOCKS:
            if kb in v2_l1:
                print(f"    v2_split_dma K={kb}: {v2_l1[kb]/ttnn_t4:.2f}x")
        for kb in K_BLOCKS:
            if kb in v4_l1:
                print(f"    v4_l1_acc    K={kb}: {v4_l1[kb]/ttnn_t4:.2f}x")

        # ---------------------------------------------------------------
        # DRAM: bf16 Accumulation (fp32_dest_acc_en=False)
        # Doubles DST capacity (bf16: 8 -> 16 tiles), enabling larger
        # subblocks and fewer acquire/release cycles.
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("DRAM: bf16 Accumulation")
        print("  4096x4096x4096, blocks 8x8, fp32_dest_acc_en=False")
        print("=" * 70)

        Mt, Kt, Nt = 128, 128, 128
        M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
        a_torch = torch.randn(M, K, dtype=torch.bfloat16)
        b_torch = torch.randn(K, N, dtype=torch.bfloat16)
        golden = (a_torch.float() @ b_torch.float()).float()

        a = ttnn.from_torch(
            a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        b = ttnn.from_torch(
            b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        cfg = {"M": M, "K": K, "N": N, "M_block": 8, "N_block": 8}

        ttnn_t5 = run_ttnn_matmul_benchmark(
            "ttnn.matmul (reference)",
            a,
            b,
            device,
            config=cfg,
        )

        v2_bf16 = sweep_k_blocks(
            "v2_split_dma bf16acc",
            make_v2,
            Kt,
            8,
            8,
            a,
            b,
            golden,
            device,
            cfg,
            strategy="manual_k_bf16acc",
            fp32_acc=False,
        )

        print(f"\n  Ratios (tt-lang / ttnn.matmul), bf16 acc:")
        for kb in K_BLOCKS:
            if kb in v2_bf16:
                print(f"    v2_split_dma K={kb}: {v2_bf16[kb]/ttnn_t5:.2f}x")

        print(f"\n{'='*70}")
        print(f"Results saved to {RESULTS_CSV}")
        print("=" * 70)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
