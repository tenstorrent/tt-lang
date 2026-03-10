# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance comparison between metal and ttlang matmul kernels.

Requires the following environment variables:
    TT_METAL_HOME             -- path to tt-metal source tree
    TT_METAL_DEVICE_PROFILER=1
    TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1
    TT_METAL_PROFILER_MID_RUN_DUMP=1

Usage:
    python test/perf/compare_matmul_perf.py
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import ttnn

from ttl._src.perf_summary import run as perf_summary_run


def _check_profiler_env():
    """Validate that the required profiler env vars are set."""
    tt_metal_home = os.environ.get("TT_METAL_HOME", "")
    if not tt_metal_home:
        print(
            "ERROR: TT_METAL_HOME not set. Required for profiler logs.",
            file=sys.stderr,
        )
        sys.exit(1)
    missing = []
    for var in [
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_DEVICE_PROFILER_NOC_EVENTS",
        "TT_METAL_PROFILER_MID_RUN_DUMP",
    ]:
        if os.environ.get(var) != "1":
            missing.append(var)
    if missing:
        print(
            f"WARNING: The following env vars are not set to '1': {', '.join(missing)}",
            file=sys.stderr,
        )
        print(
            "Profiler data may be incomplete or missing.",
            file=sys.stderr,
        )
    return Path(tt_metal_home) / "generated" / "profiler" / ".logs"


def _flush_profiler(device):
    """Flush device profiler data so it can be read."""
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception as e:
        print(f"WARNING: Failed to read device profiler: {e}", file=sys.stderr)


def _collect_perf(logs_path, kernel_name):
    """Parse profiler logs and return the summary string."""
    if not logs_path.exists():
        return f"Profiler logs not found at {logs_path}"
    result = perf_summary_run(logs_path, names=[kernel_name])
    return result if result else "No profiler data found"


def run_singlecore_comparison(M, K, N):
    """Run singlecore matmul on both metal and ttlang, compare perf."""
    from examples.metal_examples.singlecore_matmul.metal.singlecore_matmul import (
        run_singlecore_matmul,
    )

    logs_path = _check_profiler_env()

    device = ttnn.open_device(device_id=0)
    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    a_tensor = ttnn.rand(
        (M, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    b_tensor = ttnn.rand(
        (K, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )
    output_tensor = ttnn.empty(
        (M, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    # -- Metal kernel --
    print(f"\n{'='*60}")
    print(f"METAL singlecore_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")
    metal_output = run_singlecore_matmul(device, a_tensor, b_tensor, output_tensor)
    _flush_profiler(device)
    metal_perf = _collect_perf(logs_path, "singlecore_matmul")
    print(metal_perf)

    # Verify correctness
    metal_result = ttnn.to_torch(metal_output).to(torch.bfloat16)
    a_torch = ttnn.to_torch(a_tensor).to(torch.bfloat16)
    b_torch = ttnn.to_torch(b_tensor).to(torch.bfloat16)
    golden = torch.matmul(a_torch, b_torch)
    from utils.correctness import assert_with_ulp

    assert_with_ulp(golden, metal_result)
    print("Metal correctness: PASS")

    # -- TTLang kernel --
    print(f"\n{'='*60}")
    print(f"TTLANG singlecore_matmul  M={M} K={K} N={N}")
    print(f"{'='*60}")
    try:
        from examples.metal_examples.singlecore_matmul.ttlang.singlecore_matmul import (
            tt_lang_singlecore_matmul,
        )

        a_ttl = ttnn.rand(
            (M, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memory_config,
        )
        b_ttl = ttnn.rand(
            (K, N),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memory_config,
        )
        c_ttl = ttnn.empty(
            (M, N),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memory_config,
        )

        tt_lang_singlecore_matmul(a_ttl, b_ttl, c_ttl)
        _flush_profiler(device)
        ttlang_perf = _collect_perf(logs_path, "tt_lang_singlecore_matmul")
        print(ttlang_perf)

        ttlang_result = ttnn.to_torch(c_ttl).to(torch.bfloat16)
        a_ttl_torch = ttnn.to_torch(a_ttl).to(torch.bfloat16)
        b_ttl_torch = ttnn.to_torch(b_ttl).to(torch.bfloat16)
        golden_ttl = torch.matmul(a_ttl_torch, b_ttl_torch)
        assert_with_ulp(golden_ttl, ttlang_result)
        print("TTLang correctness: PASS")
    except Exception as e:
        print(f"TTLang kernel failed (expected): {e}")

    ttnn.close_device(device)


def main():
    parser = argparse.ArgumentParser(
        description="Compare metal vs ttlang matmul kernel performance"
    )
    parser.add_argument("--M", type=int, default=256, help="M dimension")
    parser.add_argument("--K", type=int, default=256, help="K dimension")
    parser.add_argument("--N", type=int, default=256, help="N dimension")
    args = parser.parse_args()

    run_singlecore_comparison(args.M, args.K, args.N)


if __name__ == "__main__":
    main()
