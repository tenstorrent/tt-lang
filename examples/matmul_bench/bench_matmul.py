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
    make_minimal_matmul,
    make_minimal_matmul_single_reader,
    make_matmul_compiler_k_loop,
    make_matmul_l1_acc,
)


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

        # v1-single-reader: all reads on one RISC core.
        out_v1a = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        single_t3a, _ = run_benchmark(
            "v1-single-reader K=8 Kblocks=16",
            make_minimal_matmul_single_reader(8, 8, 8),
            (a, b, out_v1a),
            device,
            config={**cfg, "K_block": 8, "strategy": "single_reader"},
        )
        assert_pcc(golden, ttnn.to_torch(out_v1a).float(), threshold=PCC_THRESHOLD)

        out_v1b = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        single_t3b, _ = run_benchmark(
            "v1-single-reader K=1 Kblocks=128",
            make_minimal_matmul_single_reader(8, 1, 8),
            (a, b, out_v1b),
            device,
            config={**cfg, "K_block": 1, "strategy": "single_reader"},
        )
        assert_pcc(golden, ttnn.to_torch(out_v1b).float(), threshold=PCC_THRESHOLD)

        # v2-split-dma: reads split across both RISC cores.
        out5 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t3a, _ = run_benchmark(
            "v2-split-dma K=8 Kblocks=16",
            make_minimal_matmul(8, 8, 8),
            (a, b, out5),
            device,
            config={**cfg, "K_block": 8, "strategy": "manual_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out5).float(), threshold=PCC_THRESHOLD)

        out6 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t3b, _ = run_benchmark(
            "v2-split-dma K=1 Kblocks=128",
            make_minimal_matmul(8, 1, 8),
            (a, b, out6),
            device,
            config={**cfg, "K_block": 1, "strategy": "manual_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out6).float(), threshold=PCC_THRESHOLD)

        # L1 acc: reserve once, store K times, push once.
        out7 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        l1acc_t3a, _ = run_benchmark(
            "l1_acc M=8 K=8 N=8 Kblocks=16",
            make_matmul_l1_acc(8, 8, 8),
            (a, b, out7),
            device,
            config={**cfg, "K_block": 8, "strategy": "l1_acc"},
        )
        assert_pcc(golden, ttnn.to_torch(out7).float(), threshold=PCC_THRESHOLD)

        out8 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        l1acc_t3b, _ = run_benchmark(
            "l1_acc M=8 K=1 N=8 Kblocks=128",
            make_matmul_l1_acc(8, 1, 8),
            (a, b, out8),
            device,
            config={**cfg, "K_block": 1, "strategy": "l1_acc"},
        )
        assert_pcc(golden, ttnn.to_torch(out8).float(), threshold=PCC_THRESHOLD)

        print(f"\n  Ratios (tt-lang / ttnn.matmul):")
        print(f"    v1 single K=8: {single_t3a/ttnn_t3:.2f}x")
        print(f"    v1 single K=1: {single_t3b/ttnn_t3:.2f}x")
        print(f"    v2 split  K=8: {manual_t3a/ttnn_t3:.2f}x")
        print(f"    v2 split  K=1: {manual_t3b/ttnn_t3:.2f}x")
        print(f"    l1_acc    K=8: {l1acc_t3a/ttnn_t3:.2f}x")
        print(f"    l1_acc    K=1: {l1acc_t3b/ttnn_t3:.2f}x")

        # ---------------------------------------------------------------
        # L1-Only: Compute Isolation (Section 3.1)
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("L1-Only: Compute Isolation")
        print("  2048x2048x2048 (64x64x64 tiles), blocks 8x8")
        print("=" * 70)

        Mt, Kt, Nt = 64, 64, 64
        M, K, N = Mt * TILE, Kt * TILE, Nt * TILE
        a_torch = torch.randn(M, K, dtype=torch.bfloat16)
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

        out_l1 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=l1_cfg,
        )
        manual_t4a, _ = run_benchmark(
            "manual_k M=8 K=8 N=8 Kblocks=2 (L1)",
            make_minimal_matmul(8, 8, 8),
            (a_l1, b_l1, out_l1),
            device,
            config={**cfg, "K_block": 8, "strategy": "manual_k_loop_l1"},
        )
        assert_pcc(golden, ttnn.to_torch(out_l1).float(), threshold=PCC_THRESHOLD)

        out_l1b = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=l1_cfg,
        )
        manual_t4b, _ = run_benchmark(
            "manual_k M=8 K=1 N=8 Kblocks=16 (L1)",
            make_minimal_matmul(8, 1, 8),
            (a_l1, b_l1, out_l1b),
            device,
            config={**cfg, "K_block": 1, "strategy": "manual_k_loop_l1"},
        )
        assert_pcc(golden, ttnn.to_torch(out_l1b).float(), threshold=PCC_THRESHOLD)

        # L1 acc variants.
        out_l1c = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=l1_cfg,
        )
        l1acc_t4a, _ = run_benchmark(
            "l1_acc M=8 K=8 N=8 Kblocks=2 (L1)",
            make_matmul_l1_acc(8, 8, 8),
            (a_l1, b_l1, out_l1c),
            device,
            config={**cfg, "K_block": 8, "strategy": "l1_acc_l1"},
        )
        assert_pcc(golden, ttnn.to_torch(out_l1c).float(), threshold=PCC_THRESHOLD)

        out_l1d = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=l1_cfg,
        )
        l1acc_t4b, _ = run_benchmark(
            "l1_acc M=8 K=1 N=8 Kblocks=16 (L1)",
            make_matmul_l1_acc(8, 1, 8),
            (a_l1, b_l1, out_l1d),
            device,
            config={**cfg, "K_block": 1, "strategy": "l1_acc_l1"},
        )
        assert_pcc(golden, ttnn.to_torch(out_l1d).float(), threshold=PCC_THRESHOLD)

        print(f"\n  Ratios (tt-lang / ttnn.matmul), L1-only:")
        print(f"    manual_k K=8:  {manual_t4a/ttnn_t4:.2f}x")
        print(f"    manual_k K=1:  {manual_t4b/ttnn_t4:.2f}x")
        print(f"    l1_acc   K=8:  {l1acc_t4a/ttnn_t4:.2f}x")
        print(f"    l1_acc   K=1:  {l1acc_t4b/ttnn_t4:.2f}x")

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

        out5a = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t5a, _ = run_benchmark(
            "manual_k M=8 K=8 N=8 bf16acc",
            make_minimal_matmul(8, 8, 8, fp32_acc=False),
            (a, b, out5a),
            device,
            config={**cfg, "K_block": 8, "strategy": "manual_k_bf16acc"},
        )
        assert_pcc(golden, ttnn.to_torch(out5a).float(), threshold=PCC_THRESHOLD)

        out5b = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t5b, _ = run_benchmark(
            "manual_k M=8 K=1 N=8 bf16acc",
            make_minimal_matmul(8, 1, 8, fp32_acc=False),
            (a, b, out5b),
            device,
            config={**cfg, "K_block": 1, "strategy": "manual_k_bf16acc"},
        )
        assert_pcc(golden, ttnn.to_torch(out5b).float(), threshold=PCC_THRESHOLD)

        print(f"\n  Ratios (tt-lang / ttnn.matmul), bf16 acc:")
        print(f"    manual_k K=8: {manual_t5a/ttnn_t5:.2f}x")
        print(f"    manual_k K=1: {manual_t5b/ttnn_t5:.2f}x")

        print(f"\n{'='*70}")
        print(f"Results saved to {RESULTS_CSV}")
        print("=" * 70)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
