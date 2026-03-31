# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance comparison between metal and ttlang matmul kernels.

Runs both kernel implementations on identical inputs, collects device
profiler data for each in isolation, and prints a side-by-side comparison.

Requires the following environment variables:
    TT_METAL_HOME             -- path to tt-metal source tree
    TT_METAL_DEVICE_PROFILER=1
    TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1
    TT_METAL_PROFILER_MID_RUN_DUMP=1

Usage:
    python test/perf/compare_matmul_perf.py --M 256 --K 256 --N 256
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import ttnn

from ttl._src.perf_summary import (
    ProgramSummary,
    clear_profiler_logs,
    collect_summaries,
    flush_profiler,
    run as perf_summary_run,
)
from utils.correctness import assert_with_ulp


# ---------------------------------------------------------------------------
# Profiler environment helpers
# ---------------------------------------------------------------------------

_REQUIRED_PROFILER_VARS = [
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_DEVICE_PROFILER_NOC_EVENTS",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
]


def _check_profiler_env():
    """Validate required profiler env vars and return the logs directory."""
    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    if not tt_metal_home:
        print(
            "ERROR: TT_METAL_HOME not set. Required for profiler logs.",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = [v for v in _REQUIRED_PROFILER_VARS if os.environ.get(v) != "1"]
    if missing:
        print(
            f"WARNING: env vars not set to '1': {', '.join(missing)}. "
            "Profiler data may be incomplete.",
            file=sys.stderr,
        )

    return Path(tt_metal_home) / "generated" / "profiler" / ".logs"


# ---------------------------------------------------------------------------
# Comparison formatting
# ---------------------------------------------------------------------------


def _duration_cycles(s: ProgramSummary) -> int:
    """Extract wall-clock duration in cycles from a ProgramSummary."""
    if s.min_timestamp is not None and s.max_timestamp is not None:
        return s.max_timestamp - s.min_timestamp
    return 0


def _ratio_str(a: float, b: float) -> str:
    """Format b/a as a ratio string, or 'N/A' when a is zero."""
    if a > 0:
        return f"{b / a:.2f}x"
    return "N/A"


def _print_comparison(
    metal_summaries: List[ProgramSummary],
    ttlang_summaries: List[ProgramSummary],
    freq_mhz: int,
) -> None:
    """Print a side-by-side duration comparison of metal vs ttlang kernels."""
    if not metal_summaries or not ttlang_summaries:
        return

    metal = metal_summaries[0]
    ttlang = ttlang_summaries[0]

    m_cyc = _duration_cycles(metal)
    t_cyc = _duration_cycles(ttlang)
    m_us = m_cyc / freq_mhz
    t_us = t_cyc / freq_mhz

    m_dram = metal.dram_bytes_read + metal.dram_bytes_written
    t_dram = ttlang.dram_bytes_read + ttlang.dram_bytes_written

    m_xfers = (
        metal.dram_read_count
        + metal.dram_write_count
        + metal.l1_read_count
        + metal.l1_write_count
    )
    t_xfers = (
        ttlang.dram_read_count
        + ttlang.dram_write_count
        + ttlang.l1_read_count
        + ttlang.l1_write_count
    )

    hdr = f"{'Metric':<24} {'Metal':>14} {'TTLang':>14} {'Ratio':>10}"
    sep = f"{'-'*24} {'-'*14} {'-'*14} {'-'*10}"

    print(f"\n{'='*60}")
    print("COMPARISON  (ratio = ttlang / metal)")
    print(f"{'='*60}")
    print(hdr)
    print(sep)
    print(f"{'Duration (cycles)':<24} {m_cyc:>14,} {t_cyc:>14,} {_ratio_str(m_cyc, t_cyc):>10}")
    print(f"{'Duration (us)':<24} {m_us:>14.1f} {t_us:>14.1f} {_ratio_str(m_us, t_us):>10}")
    print(f"{'DRAM traffic (B)':<24} {m_dram:>14,} {t_dram:>14,} {_ratio_str(m_dram, t_dram):>10}")
    print(f"{'NOC transfers':<24} {m_xfers:>14,} {t_xfers:>14,} {_ratio_str(m_xfers, t_xfers):>10}")
    print(f"{'Cores':<24} {len(metal.source_cores):>14} {len(ttlang.source_cores):>14}")


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------


def run_singlecore_comparison(M: int, K: int, N: int) -> None:
    """Run singlecore matmul on both metal and ttlang, compare perf."""
    from examples.metal_examples.singlecore_matmul.metal.singlecore_matmul import (
        run_singlecore_matmul,
    )

    logs_path = _check_profiler_env()

    device = ttnn.open_device(device_id=0)
    dram = ttnn.DRAM_MEMORY_CONFIG

    a_tensor = ttnn.rand(
        (M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    b_tensor = ttnn.rand(
        (K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    output_tensor = ttnn.empty(
        (M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )

    metal_summaries: List[ProgramSummary] = []
    ttlang_summaries: List[ProgramSummary] = []
    freq_mhz = 1000

    # -- Metal kernel -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"METAL singlecore_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")

    flush_profiler(device)
    clear_profiler_logs(logs_path)
    metal_output = run_singlecore_matmul(device, a_tensor, b_tensor, output_tensor)
    flush_profiler(device)

    metal_perf = perf_summary_run(logs_path, names=["singlecore_matmul"])
    _, freq_mhz, metal_summaries = collect_summaries(logs_path)
    print(metal_perf or "No profiler data found")

    metal_result = ttnn.to_torch(metal_output).to(torch.bfloat16)
    a_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    golden = torch.matmul(a_torch, b_torch)
    assert_with_ulp(golden, metal_result)
    print("Metal correctness: PASS")

    # -- TTLang kernel ------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TTLANG singlecore_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")
    try:
        from examples.metal_examples.singlecore_matmul.ttlang.singlecore_matmul import (
            tt_lang_singlecore_matmul,
        )

        c_ttl = ttnn.empty(
            (M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=dram,
        )

        flush_profiler(device)
        clear_profiler_logs(logs_path)
        tt_lang_singlecore_matmul(a_tensor, b_tensor, c_ttl)
        flush_profiler(device)

        ttlang_perf = perf_summary_run(logs_path, names=["tt_lang_singlecore_matmul"])
        _, _, ttlang_summaries = collect_summaries(logs_path)
        print(ttlang_perf or "No profiler data found")

        ttlang_result = ttnn.to_torch(c_ttl).to(torch.bfloat16)
        assert_with_ulp(golden, ttlang_result)
        print("TTLang correctness: PASS")
    except Exception as e:
        print(f"TTLang kernel failed (expected): {e}")

    # -- Side-by-side comparison --------------------------------------------
    if metal_summaries and ttlang_summaries:
        _print_comparison(metal_summaries, ttlang_summaries, freq_mhz)

    ttnn.close_device(device)


def run_multicore_comparison(M: int, K: int, N: int) -> None:
    """Run multicore matmul on metal, profile it."""
    from examples.metal_examples.multicore_matmul.metal.multicore_matmul import (
        run_multicore_matmul,
    )

    logs_path = _check_profiler_env()

    device = ttnn.open_device(device_id=0)
    dram = ttnn.DRAM_MEMORY_CONFIG

    num_output_tiles = (M * N) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
    device_core_size = device.compute_with_storage_grid_size()
    upper = ttnn.CoreCoord(device_core_size.x - 1, device_core_size.y - 1)
    device_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), upper)]
    )
    (_, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2) = (
        ttnn.split_work_to_cores(device_grid, num_output_tiles, row_wise=True)
    )

    a_tensor = ttnn.rand(
        (M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    b_tensor = ttnn.rand(
        (K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    output_tensor = ttnn.empty(
        (M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )

    metal_summaries: List[ProgramSummary] = []
    ttlang_summaries: List[ProgramSummary] = []
    freq_mhz = 1000

    # -- Metal kernel -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"METAL multicore_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")

    flush_profiler(device)
    clear_profiler_logs(logs_path)
    metal_output = run_multicore_matmul(
        device, a_tensor, b_tensor, output_tensor,
        all_cores, core_group_1, core_group_2,
        work_per_core1, work_per_core2,
    )
    flush_profiler(device)

    metal_perf = perf_summary_run(logs_path, names=["multicore_matmul"])
    _, freq_mhz, metal_summaries = collect_summaries(logs_path)
    print(metal_perf or "No profiler data found")

    metal_result = ttnn.to_torch(metal_output).to(torch.bfloat16)
    a_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    golden = torch.matmul(a_torch, b_torch)
    assert_with_ulp(golden, metal_result)
    print("Metal correctness: PASS")

    # -- TTLang kernel ------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TTLANG multicore_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")
    try:
        from examples.metal_examples.multicore_matmul.ttlang.multicore_matmul import (
            tt_lang_multicore_matmul,
        )

        output_tensor = ttnn.empty(
        (M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
        )

        flush_profiler(device)
        clear_profiler_logs(logs_path)
        tt_lang_multicore_matmul(a_tensor, b_tensor, output_tensor)
        flush_profiler(device)

        ttlang_perf = perf_summary_run(
            logs_path, names=["tt_lang_multicore_matmul"],
        )
        _, _, ttlang_summaries = collect_summaries(logs_path)
        print(ttlang_perf or "No profiler data found")

        ttlang_result = ttnn.to_torch(output_tensor).to(torch.bfloat16)
        assert_with_ulp(golden, ttlang_result)
        print("TTLang correctness: PASS")
    except Exception as e:
        print(f"TTLang kernel failed (expected): {e}")

    if metal_summaries and ttlang_summaries:
        _print_comparison(metal_summaries, ttlang_summaries, freq_mhz)

    ttnn.close_device(device)


def run_multicore_reuse_comparison(M: int, K: int, N: int) -> None:
    """Run multicore reuse matmul on metal, profile it."""
    from examples.metal_examples.multicore_reuse_matmul.metal.multicore_reuse_matmul import (
        run_multicore_reuse_matmul,
    )
    from utils.block_allocation import get_large_matmul_params

    logs_path = _check_profiler_env()

    device = ttnn.open_device(device_id=0)
    dram = ttnn.DRAM_MEMORY_CONFIG

    Mt = M // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    K_block_size = 2

    device_core_size = device.compute_with_storage_grid_size()
    block_params = get_large_matmul_params(
        Mt, Nt, device_core_size.y, device_core_size.x, K_block_size,
    )
    assert block_params.block_h != 0, (
        f"get_large_matmul_params found no solution for M={M} K={K} N={N}"
    )

    a_tensor = ttnn.rand(
        (M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    b_tensor = ttnn.rand(
        (K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    output_tensor = ttnn.empty(
        (M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )

    metal_summaries: List[ProgramSummary] = []
    ttlang_summaries: List[ProgramSummary] = []
    freq_mhz = 1000

    # -- Metal kernel -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"METAL multicore_reuse_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")

    flush_profiler(device)
    clear_profiler_logs(logs_path)
    metal_output = run_multicore_reuse_matmul(
        device, a_tensor, b_tensor, output_tensor,
        K_block_size, block_params.block_h, block_params.block_w,
        block_params.subblock_h, block_params.subblock_w,
    )
    flush_profiler(device)

    metal_perf = perf_summary_run(logs_path, names=["multicore_reuse_matmul"])
    _, freq_mhz, metal_summaries = collect_summaries(logs_path)
    print(metal_perf or "No profiler data found")

    metal_result = ttnn.to_torch(metal_output).to(torch.bfloat16)
    a_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    golden = torch.matmul(a_torch, b_torch)
    assert_with_ulp(golden, metal_result)
    print("Metal correctness: PASS")

    # -- TTLang kernel ------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TTLANG multicore_reuse_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")
    try:
        from examples.metal_examples.multicore_reuse_matmul.ttlang.multicore_reuse_matmul import (
            tt_lang_multicore_reuse_matmul,
        )

        a_host = ttnn.from_device(a_tensor)
        b_host = ttnn.from_device(b_tensor)
        c_ttl = ttnn.from_torch(
            torch.zeros(M, N, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )

        flush_profiler(device)
        clear_profiler_logs(logs_path)
        tt_lang_multicore_reuse_matmul(
            a_host, b_host, c_ttl,
            K_block_size, block_params.block_h, block_params.block_w,
        )
        flush_profiler(device)

        ttlang_perf = perf_summary_run(
            logs_path, names=["tt_lang_multicore_reuse_matmul"],
        )
        _, _, ttlang_summaries = collect_summaries(logs_path)
        print(ttlang_perf or "No profiler data found")

        ttlang_result = ttnn.to_torch(c_ttl).to(torch.bfloat16)
        assert_with_ulp(golden, ttlang_result)
        print("TTLang correctness: PASS")
    except Exception as e:
        print(f"TTLang kernel failed (expected): {e}")

    if metal_summaries and ttlang_summaries:
        _print_comparison(metal_summaries, ttlang_summaries, freq_mhz)

    ttnn.close_device(device)


def run_1d_mcast_comparison(M: int, K: int, N: int) -> None:
    """Run 1D multicast matmul on metal, profile it.

    Uses fixed blocking parameters chosen for the default (128, 16384, 256)
    problem size: block_m=4, block_n=4, block_k=2, n_blocks_per_core=2,
    subblock 2x2.  (64 cores with 2 N-blocks each.)
    """
    import importlib

    _1d_mod = importlib.import_module(
        "examples.metal_examples.1d_mcast_matmul.metal.1d_matmul_metal",
    )
    run_1d_matmul = _1d_mod.run_1d_matmul

    logs_path = _check_profiler_env()

    device = ttnn.open_device(device_id=0)
    dram = ttnn.DRAM_MEMORY_CONFIG

    block_m, block_n, block_k = 8, 8, 16
    n_blocks_per_core = 2
    subblock_h, subblock_w = 4, 2

    a_tensor = ttnn.rand(
        (M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    b_tensor = ttnn.rand(
        (K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )
    output_tensor = ttnn.empty(
        (M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=dram,
    )

    metal_summaries: List[ProgramSummary] = []
    ttlang_summaries: List[ProgramSummary] = []
    freq_mhz = 1000

    # -- Metal kernel -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"METAL 1d_mcast_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")

    flush_profiler(device)
    clear_profiler_logs(logs_path)
    metal_output = run_1d_matmul(
        device, a_tensor, b_tensor, output_tensor,
        block_m, block_n, block_k, n_blocks_per_core,
        subblock_h, subblock_w,
    )
    flush_profiler(device)

    metal_perf = perf_summary_run(logs_path, names=["1d_mcast_matmul"])
    _, freq_mhz, metal_summaries = collect_summaries(logs_path)
    print(metal_perf or "No profiler data found")

    metal_result = ttnn.to_torch(metal_output).to(torch.bfloat16)
    a_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    golden = torch.matmul(a_torch, b_torch)
    assert_with_ulp(golden, metal_result)
    print("Metal correctness: PASS")

    # -- TTLang kernel ------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TTLANG 1d_mcast_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")
    print("TTLang 1D mcast matmul not yet implemented")

    ttnn.close_device(device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_KERNEL_RUNNERS = {
    "singlecore": (run_singlecore_comparison, (640, 640, 640)),
    "multicore": (run_multicore_comparison, (640, 640, 640)),
    "multicore_reuse": (run_multicore_reuse_comparison, (640, 640, 640)),
    "1d_mcast": (run_1d_mcast_comparison, (512, 512, 61440)),
}


def main():
    parser = argparse.ArgumentParser(
        description="Compare metal vs ttlang matmul kernel performance",
    )
    parser.add_argument(
        "--kernel",
        choices=list(_KERNEL_RUNNERS.keys()) + ["all"],
        default="all",
        help="Which kernel variant to benchmark (default: all)",
    )
    parser.add_argument("--M", type=int, default=None, help="M dimension (overrides default)")
    parser.add_argument("--K", type=int, default=None, help="K dimension (overrides default)")
    parser.add_argument("--N", type=int, default=None, help="N dimension (overrides default)")
    args = parser.parse_args()

    kernels = list(_KERNEL_RUNNERS.keys()) if args.kernel == "all" else [args.kernel]

    for name in kernels:
        runner, (default_M, default_K, default_N) = _KERNEL_RUNNERS[name]
        M = args.M if args.M is not None else default_M
        K = args.K if args.K is not None else default_K
        N = args.N if args.N is not None else default_N
        runner(M, K, N)


if __name__ == "__main__":
    main()
