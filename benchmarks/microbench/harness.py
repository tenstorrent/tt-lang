# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runs a benchmark's handwritten kernels through real tt-metal dispatch and
returns warm, per-RISC device-profiler microseconds — the composed,
dataflow-buffer-inclusive cost these microbenchmarks exist to measure, as opposed
to isolated LLK primitives. runner.py layers the declarative sweep on top.
"""

import csv as _csv
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

from datetime import datetime, timezone
from pathlib import Path

import torch
import ttnn

from benchmarks.microbench import profiler

TILE = 32

# name -> (ttnn dtype, torch dtype, bytes/datum)
DTYPES = {
    "bf16": (ttnn.bfloat16, torch.bfloat16, 2),
    "fp32": (ttnn.float32, torch.float32, 4),
}
FIDELITY = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi4": ttnn.MathFidelity.HiFi4,
}


def dst_capacity(dtype, full_sync, fp32_dest_acc):
    """DST tile capacity: 16/8 (16-bit dest, full-sync/default); 8/4 for fp32 dest."""
    fp32_dst = (dtype == "fp32") or fp32_dest_acc
    if fp32_dst:
        return 8 if full_sync else 4
    return 16 if full_sync else 8


def dst_subblock(rows, cols, capacity):
    """Largest (sub_rows, sub_cols) dividing (rows, cols) with product <= capacity.

    Mirrors the compiler's computeMultiDimSubblockSizes for matmul parallel
    dims: maximize the subblock tile product, breaking ties toward the larger
    inner (cols) dimension. K accumulates in-place, so the budget is the full
    DST capacity. Returns the subblock the compiler would pick for an
    `rows x cols` matmul output, so the benchmark's reuse factor matches.
    """

    def divisors(value):
        return [d for d in range(value, 0, -1) if value % d == 0]

    best, best_product = (1, 1), 1
    for sub_rows in divisors(rows):
        for sub_cols in divisors(cols):
            product = sub_rows * sub_cols
            if product > capacity:
                continue
            prefers_inner = product == best_product and (
                sub_cols > best[1] or (sub_cols == best[1] and sub_rows > best[0])
            )
            if product > best_product or prefers_inner:
                best, best_product = (sub_rows, sub_cols), product
    return best


def single_core():
    core = ttnn.CoreCoord(0, 0)
    return core, ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def dfb(buffer_index, data_format, page_size, core_grid, pages):
    fmt = ttnn.CBFormatDescriptor(
        buffer_index=buffer_index, data_format=data_format, page_size=page_size
    )
    return ttnn.CBDescriptor(
        total_size=pages * page_size, core_ranges=core_grid, format_descriptors=[fmt]
    )


def accessor_args(*tensors):
    """Concatenated TensorAccessor compile-time args for one or more tensors."""
    args = []
    for tensor in tensors:
        args += list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
    return args


def file_kernel(source, core_grid, compile_time_args, runtime_args, config):
    return ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=compile_time_args,
        runtime_args=runtime_args,
        config=config,
    )


def compute_config(fidelity="hifi4", fp32_dest_acc=False, full_sync=False):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = FIDELITY[fidelity]
    cfg.fp32_dest_acc_en = fp32_dest_acc
    cfg.dst_full_sync_en = full_sync
    cfg.math_approx_mode = False
    return cfg


def dispatch(device, io_tensors, kernels, dfbs, zone_name, warmup=1):
    """Run a generic_op program and return (output_tensor, profiler zone summary).

    `warmup` dispatches run first to warm the kernel-binary / instruction / data
    caches; their profiler zones are flushed and discarded so the returned zone
    reflects only the measured (warm) run.
    """
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=dfbs)
    csv_path = profiler.find_profiler_csv()
    for _ in range(warmup):
        ttnn.generic_op(io_tensors, program)
        ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)  # flush warmup zones...
    if csv_path.exists():
        csv_path.unlink()  # ...and discard them
    output = ttnn.generic_op(io_tensors, program)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    return output, profiler.summarize_zone(csv_path, zone_name)


def write_csv(rows, base_csv, fields, *tag_parts):
    """Write rows to a dated CSV `<stem>_<tag_parts>_<UTCstamp>.csv`; return the path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(base_csv)
    tag = "_".join(str(p) for p in tag_parts if p)
    out_csv = base.with_name(f"{base.stem}_{tag}_{stamp}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as file:
        writer = _csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def zone_fields(zone, pcc):
    """The result fields common to every benchmark, from a profiler zone summary."""
    return {
        "arch": zone["arch"],
        "freq_mhz": zone["freq_mhz"],
        "trisc_max_us": zone["trisc_max_us"],
        "unpack_us": zone["unpack_us"],
        "math_us": zone["math_us"],
        "pack_us": zone["pack_us"],
        "noc_active_in_zone": int(zone["noc_active_in_zone"]),
        "pcc": round(pcc, 6),
    }
