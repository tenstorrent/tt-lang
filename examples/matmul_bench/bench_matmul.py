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

from minimal_matmul import make_minimal_matmul, make_matmul_compiler_k_loop


def main():
    device = ttnn.open_device(device_id=0)
    try:
        print("=" * 70)
        print("MATMUL PERFORMANCE BENCHMARK")
        print("=" * 70)
        print(f"  Git: {get_git_sha()}")
        print(f"  Results: {RESULTS_CSV}")

        # ---------------------------------------------------------------
        # Test 1: Small problem, output fits in DST, K > 1
        # Both strategies should work; compiler_k avoids copy_tile overhead.
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("Test 1: 128x128x128 (4x4x4 tiles), blocks 2x2, K_block=2")
        print("  Output 2x2=4 tiles fits in bf16 DST (8). K=2 inner tiles.")
        print("=" * 70)

        Mt, Kt, Nt = 4, 4, 4
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

        cfg = {"M": M, "K": K, "N": N, "M_block": 2, "N_block": 2}

        ttnn_t1 = run_ttnn_matmul_benchmark(
            "ttnn.matmul (reference)",
            a,
            b,
            device,
            config=cfg,
        )

        out1 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t1, _ = run_benchmark(
            "manual_k M=2 K=2 N=2 Kblocks=2",
            make_minimal_matmul(2, 2, 2),
            (a, b, out1),
            device,
            config={**cfg, "K_block": 2, "strategy": "manual_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out1).float(), threshold=0.99)

        out2 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        compiler_t1, _ = run_benchmark(
            "compiler_k M=2 N=2 Kfull=4",
            make_matmul_compiler_k_loop(2, 2),
            (a, b, out2),
            device,
            config={**cfg, "K_block": 4, "strategy": "compiler_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out2).float(), threshold=0.99)

        print(f"\n  Ratios (tt-lang / ttnn.matmul):")
        print(f"    manual_k:   {manual_t1/ttnn_t1:.2f}x")
        print(f"    compiler_k: {compiler_t1/ttnn_t1:.2f}x")

        # ---------------------------------------------------------------
        # Test 2: Output > DST capacity, K > 1
        # This is the target for the L1 acc plan. Currently both use
        # per-tile DST acc (no subblocking). After implementation,
        # compiler_k will use hybrid subblocking + L1 acc.
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("Test 2: 128x128x128 (4x4x4 tiles), blocks 4x4, K_block=2")
        print("  Output 4x4=16 tiles > bf16 DST (8). K=2 inner tiles.")
        print("  Target for L1 acc: subblocked 2x2 within DST, L1 across K.")
        print("=" * 70)

        cfg = {"M": M, "K": K, "N": N, "M_block": 4, "N_block": 4}

        ttnn_t2 = run_ttnn_matmul_benchmark(
            "ttnn.matmul (reference)",
            a,
            b,
            device,
            config=cfg,
        )

        out3 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t2, _ = run_benchmark(
            "manual_k M=4 K=2 N=4 Kblocks=2",
            make_minimal_matmul(4, 2, 4),
            (a, b, out3),
            device,
            config={**cfg, "K_block": 2, "strategy": "manual_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out3).float(), threshold=0.99)

        out4 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        compiler_t2, _ = run_benchmark(
            "compiler_k M=4 N=4 Kfull=4",
            make_matmul_compiler_k_loop(4, 4),
            (a, b, out4),
            device,
            config={**cfg, "K_block": 4, "strategy": "compiler_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out4).float(), threshold=0.99)

        print(f"\n  Ratios (tt-lang / ttnn.matmul):")
        print(f"    manual_k:   {manual_t2/ttnn_t2:.2f}x")
        print(f"    compiler_k: {compiler_t2/ttnn_t2:.2f}x")

        # ---------------------------------------------------------------
        # Test 3: Benchmark shape 4096x4096x4096
        # Only manual_k works at this scale (full K=128 overflows L1).
        # Compare K_block=8 (matching tt-metal) vs K_block=1 (streaming).
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("Test 3: 4096x4096x4096 (128x128x128 tiles), blocks 8x8")
        print("  tt-metal benchmark reference shape.")
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

        out5 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t3a, _ = run_benchmark(
            "manual_k M=8 K=8 N=8 Kblocks=16",
            make_minimal_matmul(8, 8, 8),
            (a, b, out5),
            device,
            config={**cfg, "K_block": 8, "strategy": "manual_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out5).float(), threshold=0.99)

        out6 = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        manual_t3b, _ = run_benchmark(
            "manual_k M=8 K=1 N=8 Kblocks=128",
            make_minimal_matmul(8, 1, 8),
            (a, b, out6),
            device,
            config={**cfg, "K_block": 1, "strategy": "manual_k_loop"},
        )
        assert_pcc(golden, ttnn.to_torch(out6).float(), threshold=0.99)

        print(f"\n  Ratios (tt-lang / ttnn.matmul):")
        print(f"    manual_k K=8: {manual_t3a/ttnn_t3:.2f}x")
        print(f"    manual_k K=1: {manual_t3b/ttnn_t3:.2f}x")

        # ---------------------------------------------------------------
        # Test 4: L1-only (no DRAM DMA), isolate compute kernel cost.
        # 512x512x512 in L1 (fits with interleaved layout).
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("Test 4: L1-only 2048x2048x2048 (64x64x64 tiles), blocks 8x8")
        print("  Inputs/outputs in L1. Isolates compute kernel cost.")
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
        assert_pcc(golden, ttnn.to_torch(out_l1).float(), threshold=0.99)

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
        assert_pcc(golden, ttnn.to_torch(out_l1b).float(), threshold=0.99)

        print(f"\n  Ratios (tt-lang / ttnn.matmul), L1-only:")
        print(f"    manual_k K=8: {manual_t4a/ttnn_t4:.2f}x")
        print(f"    manual_k K=1: {manual_t4b/ttnn_t4:.2f}x")

        # ---------------------------------------------------------------
        # Test 5: bf16 accumulation (fp32_dest_acc_en=False)
        # Doubles DST capacity (bf16: 8 -> 16 tiles), enabling larger
        # subblocks and fewer acquire/release cycles.
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("Test 5: 4096x4096x4096, blocks 8x8, bf16 accumulation")
        print("  fp32_dest_acc_en=False: DST capacity 16 (bf16 double-buf).")
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
        assert_pcc(golden, ttnn.to_torch(out5a).float(), threshold=0.99)

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
        assert_pcc(golden, ttnn.to_torch(out5b).float(), threshold=0.99)

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
