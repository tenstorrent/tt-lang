# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unary SFPU-op subblock sweep (MB5 variant): optimal DST chunk vs total tiles.

Drives compute_unary.cpp (op selected by --op, default exp) over `tiles` resident
tiles, subblocked by `sub` = the DST chunk moved per tile_regs_acquire. Unlike the
pack/unpack probe this sweeps *both* the total tiles and the chunk, so the optimal
chunk as a function of size can be read off directly for a real SFPU op. The input
is resident (CB hop outside the measured zone) and the op init is hoisted, so each
acquire is purely copy_tile -> SFPU op -> pack on its chunk.

    python -m benchmarks.microbench.mb5.subblock_unary_op_sweep \
        --tiles 8,16,32,64,128,256 --sub 1,2,4,8 --op exp
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

from pathlib import Path

import torch
import ttnn

from benchmarks.microbench import harness
from benchmarks.microbench.harness import TILE
from benchmarks.microbench.runner import DFB, MicroBenchmark, Param, Tensor

KERNELS = Path("benchmarks/microbench/kernels")
COMPUTE_KERNEL = str(KERNELS / "compute" / "compute_unary.cpp")
READER_KERNEL = str(KERNELS / "common" / "seed_reader.cpp")
WRITER_KERNEL = str(KERNELS / "common" / "drain_writer.cpp")

OP_IDS = {"copy": 0, "exp": 1, "gelu": 2, "recip": 3, "sqrt": 4, "rsqrt": 5}


def _ref(op, x):
    if op == "copy":
        return x
    if op == "exp":
        return torch.exp(x)
    if op == "gelu":
        return torch.nn.functional.gelu(x, approximate="tanh")
    if op == "recip":
        return torch.reciprocal(x)
    if op == "sqrt":
        return torch.sqrt(x)
    if op == "rsqrt":
        return torch.rsqrt(x)
    raise ValueError(f"unknown op {op!r}")


class UnaryOpSubblockSweep(MicroBenchmark):
    NAME = "unary SFPU op subblock sweep"
    ZONE = "compute_op_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/subblock_unary_op.csv"
    STRATEGIES = ("",)
    PER_UNIT = "iters"
    CSV_TAG = ("dtype", "op", "full_sync")
    EXTRA_COLUMNS = ("dst_chunk", "acquires", "compiler_pick")
    PARAMS = (
        Param("tiles", "8,16,32,64,128", sweep=True, help="total tiles (resident)"),
        Param("sub", "1,2,4,8", sweep=True, help="DST chunk per acquire"),
        Param("op", "exp", choices=tuple(OP_IDS), help="unary SFPU op"),
        Param("iters", "128", help="measured iterations"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("init_hoist", "1", help="hoist op init out of the loop (0/1)"),
        Param("full_sync", "0", sweep=True, help="dst_full_sync_en (0/1)"),
        Param("fp32_dest_acc", "0", sweep=True, help="fp32_dest_acc_en (0/1)"),
    )
    # x positive so recip/sqrt/rsqrt/exp are well-defined and exp does not overflow.
    INPUTS = (
        Tensor("x", lambda cfg: (TILE, cfg["tiles"] * TILE), scale=0.1, offset=1.0),
    )
    OUTPUTS = (
        Tensor("out", lambda cfg: (TILE, cfg["tiles"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["tiles"]),  # input (resident)
        DFB(16, lambda cfg: cfg["tiles"]),  # output (drained once)
    )

    def _cap(self, cfg):
        full_sync, fp32 = bool(int(cfg["full_sync"])), bool(int(cfg["fp32_dest_acc"]))
        return harness.dst_capacity(cfg["dtype"], full_sync, fp32)

    def legal(self, cfg, strategy):
        sub = cfg["sub"]
        return 0 < sub <= self._cap(cfg) and sub <= cfg["tiles"]

    def min_pcc(self, cfg, strategy):
        return 0.98 if cfg["op"] == "gelu" else 0.99

    def extra_columns(self, cfg, strategy):
        sub, tiles = cfg["sub"], cfg["tiles"]
        # 1D flat array: the max-DST chunk is the whole DST capacity.
        return {
            "dst_chunk": sub,
            "acquires": -(-tiles // sub),
            "compiler_pick": int(sub == min(self._cap(cfg), tiles)),
        }

    def summary(self, cfg, by_strategy):
        row = next(iter(by_strategy.values()))
        per_iter = row["trisc_max_us_per_iters"]
        per_iter = "n/a" if per_iter is None else f"{per_iter:.4f}"
        star = "  <== max-DST" if row["compiler_pick"] else ""
        return (
            f"op={cfg['op']:<6} tiles={cfg['tiles']:>3} sub={cfg['sub']} "
            f"acq={row['acquires']:>3} fs={cfg['full_sync']} | "
            f"unpack={row['unpack_us']} math={row['math_us']} pack={row['pack_us']} "
            f"trisc_max={row['trisc_max_us']} per_iter={per_iter} µs "
            f"pcc={row['pcc']}{star}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        tiles, iters, sub = cfg["tiles"], cfg["iters"], cfg["sub"]
        full_sync, fp32 = bool(int(cfg["full_sync"])), bool(int(cfg["fp32_dest_acc"]))
        init_hoist = int(cfg["init_hoist"])
        tensors = ctx.tensors
        compute = harness.compute_config(fp32_dest_acc=fp32, full_sync=full_sync)
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["x"]),
                [(ctx.core, [tensors["x"].buffer_address(), tiles])],
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
                COMPUTE_KERNEL,
                ctx.grid,
                # op, tiles, iters, cap(=chunk), init_hoist
                [OP_IDS[cfg["op"]], tiles, iters, sub, init_hoist],
                [],
                compute,
            ),
        ]
        ref = _ref(cfg["op"], ctx.torch["x"].float())
        return kernels, ref


if __name__ == "__main__":
    UnaryOpSubblockSweep().main()
