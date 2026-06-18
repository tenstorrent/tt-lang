# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB2 — accumulation: DST-resident vs L1-pack (declared on the MicroBenchmark runner).

out = initial + sum of `iters` contributions on an acc_tiles-wide accumulator,
two strategies: DST-resident (binary_dest_reuse, pack once) and L1-pack (per-step
pack_reconfig_l1_acc). `--source l1|dram` selects contribution residency
(re-read one L1 block vs stream one block per iteration). See runner.py.

    python -m benchmarks.microbench.acc_sweep --acc-tiles 1,2,4 --iters 1,2,4,8,16 --source l1
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
DST_KERNEL = str(KERNELS / "acc_dst.cpp")
L1_KERNEL = str(KERNELS / "acc_l1.cpp")
READER_KERNEL = str(KERNELS / "acc_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")


def _src_blocks(cfg):
    # initial block + delta blocks (1 re-read block for l1, `iters` streamed for dram)
    return 1 + (cfg["iters"] if cfg["source"] == "dram" else 1)


class Accumulation(MicroBenchmark):
    NAME = "accumulation DST vs L1-pack"
    ZONE = "acc_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/accumulation.csv"
    STRATEGIES = ("dst", "l1")
    PER_UNIT = "iters"
    CSV_TAG = ("dtype", "source")
    PARAMS = (
        Param("acc_tiles", "1,2,4", sweep=True, help="accumulator tiles"),
        Param("iters", "1,2,4,8,16", sweep=True, help="contribution count"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("source", "l1", choices=("l1", "dram")),
        Param("block_count", str(DEFAULT_BLOCK_COUNT), sweep=True, help="contribution/output DFB block count"),
        Param("full_sync", False),
        Param("fp32_dest_acc", False),
    )
    INPUTS = (
        Tensor("src", lambda cfg: (TILE, _src_blocks(cfg) * cfg["acc_tiles"] * TILE)),
    )
    OUTPUTS = (
        Tensor("out", lambda cfg: (TILE, cfg["acc_tiles"] * TILE), init="empty"),
    )
    # dfb_init holds the seed (read once); dfb_delta/dfb_out hold `block_count`
    # blocks, so the reader can prefetch that many contribution blocks ahead.
    DFBS = (
        DFB(0, lambda cfg: cfg["acc_tiles"]),  # dfb_init
        DFB(1, lambda cfg: cfg["block_count"] * cfg["acc_tiles"]),  # dfb_delta
        DFB(16, lambda cfg: cfg["block_count"] * cfg["acc_tiles"]),  # dfb_out
    )

    def legal(self, cfg, strategy):
        if strategy == "dst":
            cap = harness.dst_capacity(
                cfg["dtype"], cfg["full_sync"], cfg["fp32_dest_acc"]
            )
            return cfg["acc_tiles"] <= cap
        return True

    def build(self, ctx):
        cfg = ctx.cfg
        acc_tiles, iters = cfg["acc_tiles"], cfg["iters"]
        cap = harness.dst_capacity(cfg["dtype"], cfg["full_sync"], cfg["fp32_dest_acc"])
        groups = iters if cfg["source"] == "dram" else 1
        reuse = 0 if cfg["source"] == "dram" else 1
        tensors = ctx.tensors

        # out = initial + sum of `iters` deltas, sliced from the concatenated src.
        src = ctx.torch["src"].float()
        cols = acc_tiles * TILE
        initial, deltas = src[:, :cols], src[:, cols:]
        if cfg["source"] == "dram":
            ref = initial + deltas.reshape(TILE, iters, acc_tiles, TILE).sum(
                dim=1
            ).reshape(TILE, cols)
        else:
            ref = initial + iters * deltas

        compute = harness.compute_config(
            fp32_dest_acc=cfg["fp32_dest_acc"], full_sync=cfg["full_sync"]
        )
        compute_cta = (
            [acc_tiles, iters, reuse]
            if ctx.strategy == "dst"
            else [acc_tiles, iters, cap, reuse]
        )
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["src"]),
                [(ctx.core, [tensors["src"].buffer_address(), acc_tiles, groups])],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["out"]),
                [(ctx.core, [tensors["out"].buffer_address(), acc_tiles])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                DST_KERNEL if ctx.strategy == "dst" else L1_KERNEL,
                ctx.grid,
                compute_cta,
                [],
                compute,
            ),
        ]
        return kernels, ref


if __name__ == "__main__":
    Accumulation().main()
