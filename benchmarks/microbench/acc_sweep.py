# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB2 -- accumulation: DST-resident vs L1-pack (declared on the MicroBenchmark runner).

out = initial + sum of `iters` contributions on an acc_tiles-wide accumulator,
two strategies: DST-resident (binary_dest_reuse, pack once) and L1-pack (per-step
pack_reconfig_l1_acc). `--source l1|dram` selects contribution residency
(re-read one L1 block vs stream one block per iteration). `--expr add|mul|gelu`
selects the per-iteration contribution expression. See runner.py.

    python -m benchmarks.microbench.acc_sweep --acc-tiles 1,2,4 --iters 1,2,4,8,16 --source l1
    python -m benchmarks.microbench.acc_sweep --expr mul --source dram
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

from pathlib import Path

import torch
import ttnn

from benchmarks.microbench import harness
from benchmarks.microbench.harness import DEFAULT_BLOCK_COUNT, TILE
from benchmarks.microbench.runner import DFB, MicroBenchmark, Param, Tensor

KERNELS = Path("benchmarks/microbench/kernels")
DST_KERNEL = str(KERNELS / "acc_dst.cpp")
L1_KERNEL = str(KERNELS / "acc_l1.cpp")
READER_KERNEL = str(KERNELS / "acc_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")

EXPR_CHOICES = ("add", "mul", "gelu")
EXPR_IDS = {expr: expr_id for expr_id, expr in enumerate(EXPR_CHOICES)}


def _src_blocks(cfg):
    # initial block + delta blocks (1 re-read block for l1, `iters` streamed for dram)
    return 1 + (cfg["iters"] if cfg["source"] == "dram" else 1)


class Accumulation(MicroBenchmark):
    NAME = "accumulation DST vs L1-pack"
    ZONE = "acc_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/accumulation.csv"
    STRATEGIES = ("dst", "l1")
    PER_UNIT = "iters"
    CSV_TAG = ("dtype", "source", "expr", "full_sync")
    PARAMS = (
        Param("acc_tiles", "1,2,4", sweep=True, help="accumulator tiles"),
        Param("iters", "1,2,4,8,16", sweep=True, help="contribution count"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("source", "l1", choices=("l1", "dram")),
        Param(
            "expr",
            "add",
            choices=EXPR_CHOICES,
            help="per-iteration contribution: add, mul, or gelu",
        ),
        Param(
            "block_count",
            str(DEFAULT_BLOCK_COUNT),
            sweep=True,
            help="contribution/output DFB block count",
        ),
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
            # expr=mul has no DST-resident form: mul_tiles overwrites the dest
            # tile (no accumulation) and no FPU op adds two DST tiles, so a
            # product cannot accumulate in DST in place. L1-pack handles it via
            # packer accumulation. See acc_dst.cpp for the tt-metal reference.
            if cfg["expr"] == "mul":
                return False
            cap = harness.dst_capacity(
                cfg["dtype"], cfg["full_sync"], cfg["fp32_dest_acc"]
            )
            # gelu computes each contribution in a temporary DST slot before
            # adding it into the accumulator, so it needs two slots per tile.
            if cfg["expr"] == "gelu":
                return 2 * cfg["acc_tiles"] <= cap
            return cfg["acc_tiles"] <= cap
        return True

    def min_pcc(self, cfg, strategy):
        # Fast SFPU GELU is the tanh approximation; match MB3's tolerance.
        return 0.98 if cfg["expr"] == "gelu" else self.MIN_PCC

    def build(self, ctx):
        cfg = ctx.cfg
        acc_tiles, iters = cfg["acc_tiles"], cfg["iters"]
        cap = harness.dst_capacity(cfg["dtype"], cfg["full_sync"], cfg["fp32_dest_acc"])
        groups = iters if cfg["source"] == "dram" else 1
        reuse = 0 if cfg["source"] == "dram" else 1
        expr_id = EXPR_IDS[cfg["expr"]]
        tensors = ctx.tensors

        # Slice the concatenated source into the seed and contribution tiles.
        src = ctx.torch["src"].float()
        cols = acc_tiles * TILE
        initial, deltas = src[:, :cols], src[:, cols:]
        if cfg["source"] == "dram":
            deltas_by_iter = deltas.reshape(TILE, iters, acc_tiles, TILE)
        else:
            deltas_by_iter = deltas.reshape(TILE, 1, acc_tiles, TILE).expand(
                -1, iters, -1, -1
            )
        if cfg["expr"] == "add":
            contributions = deltas_by_iter
        elif cfg["expr"] == "mul":
            contributions = deltas_by_iter * deltas_by_iter
        else:
            contributions = torch.nn.functional.gelu(deltas_by_iter, approximate="tanh")
        ref = initial + contributions.sum(dim=1).reshape(TILE, cols)

        compute = harness.compute_config(
            fp32_dest_acc=cfg["fp32_dest_acc"], full_sync=cfg["full_sync"]
        )
        compute_cta = (
            [acc_tiles, iters, reuse, expr_id]
            if ctx.strategy == "dst"
            else [acc_tiles, iters, cap, reuse, expr_id]
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
