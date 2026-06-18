# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB1 — pack/unpack probe (declared on the MicroBenchmark runner).

Zero-compute DFB->DFB round-trip on one compute core: per iteration, pack `tiles`
tiles to L1 and unpack them back, isolating the per-tile pack/unpack + DFB-sync
cost. fit.py regresses the per-iteration times vs `tiles` into fixed + per-tile.
Sweep dtype/full_sync/fp32_dest_acc for the config matrix. See runner.py.

    python -m benchmarks.microbench.sweep --tiles 1,2,4,8,16 --iters 128
    python -m benchmarks.microbench.sweep --tiles 1,2,4,8 --dtype bf16,fp32 --full-sync 0,1
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

from pathlib import Path

import ttnn

from benchmarks.microbench import harness
from benchmarks.microbench.harness import DEFAULT_BLOCK_COUNT, TILE
from benchmarks.microbench.runner import DFB, MicroBenchmark, Param, Tensor

KERNELS = Path("benchmarks/microbench/kernels")
COMPUTE_KERNEL = str(KERNELS / "passthrough_compute.cpp")
READER_KERNEL = str(KERNELS / "seed_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")


class PackUnpackProbe(MicroBenchmark):
    NAME = "pack/unpack probe"
    ZONE = "pack_unpack_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/pack_unpack.csv"
    STRATEGIES = ("",)
    PER_UNIT = "iters"
    PARAMS = (
        Param("tiles", "1", sweep=True, help="tiles per iteration"),
        Param("iters", "128", help="measured iterations"),
        Param("dtype", "bf16", sweep=True, help="bf16 and/or fp32"),
        Param("full_sync", "0", sweep=True, help="dst_full_sync_en (0/1)"),
        Param("fp32_dest_acc", "0", sweep=True, help="fp32_dest_acc_en (0/1)"),
        Param(
            "block_count", str(DEFAULT_BLOCK_COUNT), sweep=True, help="DFB block count"
        ),
    )
    INPUTS = (Tensor("src", lambda cfg: (TILE, cfg["tiles"] * TILE)),)
    OUTPUTS = (Tensor("out", lambda cfg: (TILE, cfg["tiles"] * TILE), init="empty"),)
    DFBS = (
        DFB(0, lambda cfg: cfg["block_count"] * cfg["tiles"]),  # dfb_in
        DFB(1, lambda cfg: cfg["block_count"] * cfg["tiles"]),  # dfb_loop
        DFB(16, lambda cfg: cfg["block_count"] * cfg["tiles"]),  # dfb_out
    )

    def build(self, ctx):
        cfg = ctx.cfg
        tiles, iters = cfg["tiles"], cfg["iters"]
        # 0/1 flags; int() so a stray non-numeric string fails loudly instead of
        # silently becoming True (bool("false") is True).
        full_sync, fp32 = bool(int(cfg["full_sync"])), bool(int(cfg["fp32_dest_acc"]))
        cap = harness.dst_capacity(cfg["dtype"], full_sync, fp32)
        tensors = ctx.tensors
        compute = harness.compute_config(fp32_dest_acc=fp32, full_sync=full_sync)
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["src"]),
                [(ctx.core, [tensors["src"].buffer_address(), tiles])],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["out"]),
                [(ctx.core, [tensors["out"].buffer_address(), tiles])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                COMPUTE_KERNEL, ctx.grid, [tiles, iters, cap], [], compute
            ),
        ]
        return kernels, ctx.torch["src"]


if __name__ == "__main__":
    PackUnpackProbe().main()
