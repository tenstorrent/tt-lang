# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers for hardware benchmarks."""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
import ttnn
from utils.correctness import assert_pcc


def create_benchmark_arg_parser(
    description,
    *,
    default_csv,
    default_warmup=3,
    default_runs=10,
    default_seed=2026,
    default_device_id=0,
):
    """Create a parser with standard hardware benchmark controls."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--warmup", type=int, default=default_warmup)
    parser.add_argument("--runs", type=int, default=default_runs)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--device-id", type=int, default=default_device_id)
    parser.add_argument("--csv", type=Path, default=Path(default_csv))
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=os.getenv("TTLANG_COMPILE_ONLY") == "1",
    )
    parser.add_argument("--no-csv", action="store_true")
    return parser


def to_device(
    device,
    tensor,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def measure_pcc(golden, actual):
    """Return PCC for reporting; use `assert_pcc` for correctness checks."""
    golden_flat = torch.nan_to_num(golden.float().flatten())
    actual_flat = torch.nan_to_num(actual.float().flatten())

    if torch.any(golden_flat.bool()).item() != torch.any(actual_flat.bool()).item():
        return 0.0

    if torch.equal(golden_flat, actual_flat):
        return 1.0

    value = torch.corrcoef(torch.stack([golden_flat, actual_flat]))[0, 1].item()
    if not math.isfinite(value):
        return 1.0
    return value


def time_runs(thunk, device, *, warmup=3, runs=10, cleanup=lambda result: None):
    """Return steady-state wall time using the PR 661 timing convention.

    Warmup runs execute first and the device is synchronized once. Timed runs
    are then enqueued back-to-back and the device is synchronized once after the
    loop. The return value is the mean wall time per timed run.
    """
    for _ in range(warmup):
        cleanup(thunk())
    ttnn.synchronize_device(device)

    start = time.perf_counter()
    for _ in range(runs):
        cleanup(thunk())
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) / runs


def write_csv(path, fields, row):
    output_path = Path(path)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
