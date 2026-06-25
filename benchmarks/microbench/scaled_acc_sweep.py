# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Scaled-accumulate matmul subblock sweep (from test_matmul_scaled_acc.py).

`out = scale * acc + (a @ b)` -- a real matmul with a pre-seeded accumulator (not
an epilogue): mul_tiles computes scale*acc into the output subblock's DST slots,
then the Kt-step matmul_block loop accumulates a @ b onto those same slots. The
scale*acc seed shares the matmul accumulator, so it needs no DST scratch. Holds
Mt/Nt/Kt fixed and force-sweeps the output subblock (sub_mt, sub_nt); the
`compiler_pick` column models the conservative pick a heuristic would make if it
reserved ~cap/2 for the scale*acc term, so the sweep shows whether the larger
subblock (lower reuse) it would skip is actually faster.

    python -m benchmarks.microbench.scaled_acc_sweep --mt 8 --nt 8 --kt 4
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
COMPUTE_KERNEL = str(KERNELS / "scaled_acc_compute.cpp")
READER_KERNEL = str(KERNELS / "scaled_acc_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")


class ScaledAccSweep(MicroBenchmark):
    NAME = "scaled-acc matmul subblock sweep"
    ZONE = "scaled_acc_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/scaled_acc.csv"
    STRATEGIES = ("",)
    CSV_TAG = ("dtype", "full_sync")
    EXTRA_COLUMNS = ("reuse", "compiler_pick")
    PARAMS = (
        Param("mt", "8", help="output rows (tiles)"),
        Param("nt", "8", help="output cols (tiles)"),
        Param("kt", "4", help="K-depth (tiles)"),
        Param("sub_mt", "1,2,4,8", sweep=True, help="subblock rows"),
        Param("sub_nt", "1,2,4,8", sweep=True, help="subblock cols"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("full_sync", False),
        Param("block_count", str(DEFAULT_BLOCK_COUNT), sweep=True),
    )
    INPUTS = (
        Tensor("a", lambda cfg: (cfg["mt"] * TILE, cfg["kt"] * TILE), scale=0.1),
        Tensor("b", lambda cfg: (cfg["kt"] * TILE, cfg["nt"] * TILE), scale=0.1),
        Tensor("scale", lambda cfg: (cfg["mt"] * TILE, cfg["nt"] * TILE), scale=0.1),
        Tensor("acc", lambda cfg: (cfg["mt"] * TILE, cfg["nt"] * TILE), scale=0.1),
    )
    OUTPUTS = (
        Tensor("out", lambda cfg: (cfg["mt"] * TILE, cfg["nt"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["mt"] * cfg["kt"]),  # a
        DFB(1, lambda cfg: cfg["kt"] * cfg["nt"]),  # b
        DFB(2, lambda cfg: cfg["mt"] * cfg["nt"]),  # scale
        DFB(3, lambda cfg: cfg["mt"] * cfg["nt"]),  # acc
        DFB(16, lambda cfg: cfg["block_count"] * cfg["mt"] * cfg["nt"]),  # out
    )
    # the mul seed + matmul accumulate keeps the precision loose (fused chain).
    MIN_PCC = 0.99

    def _cap(self, cfg):
        return harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )

    def legal(self, cfg, strategy):
        sm, sn = cfg["sub_mt"], cfg["sub_nt"]
        if cfg["mt"] % sm or cfg["nt"] % sn:
            return False
        return sm * sn <= self._cap(cfg)

    def extra_columns(self, cfg, strategy):
        reuse = (cfg["mt"] // cfg["sub_mt"]) * (cfg["nt"] // cfg["sub_nt"])
        # Model a heuristic that reserves ~half the DST for the scale*acc seed
        # (effective budget ~cap/2), to flag the conservative pick it would make.
        pick = harness.dst_subblock(cfg["mt"], cfg["nt"], self._cap(cfg) // 2)
        return {"reuse": reuse, "compiler_pick": int((cfg["sub_mt"], cfg["sub_nt"]) == pick)}

    def summary(self, cfg, by_strategy):
        row = next(iter(by_strategy.values()))
        star = "  <== compiler-pick (est.)" if row["compiler_pick"] else ""
        return (
            f"mt={cfg['mt']} nt={cfg['nt']} sub=({cfg['sub_mt']},{cfg['sub_nt']}) "
            f"reuse={row['reuse']} | trisc_max={row['trisc_max_us']} µs "
            f"unpack={row['unpack_us']} math={row['math_us']} pack={row['pack_us']} "
            f"| pcc={row['pcc']}{star}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        mt, nt, kt = cfg["mt"], cfg["nt"], cfg["kt"]
        tensors = ctx.tensors
        compute = harness.compute_config(
            "hifi4",
            fp32_dest_acc=cfg["dtype"] == "fp32",
            full_sync=cfg["full_sync"],
        )
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(
                    tensors["a"], tensors["b"], tensors["scale"], tensors["acc"]
                ),
                [
                    (
                        ctx.core,
                        [
                            tensors["a"].buffer_address(),
                            tensors["b"].buffer_address(),
                            tensors["scale"].buffer_address(),
                            tensors["acc"].buffer_address(),
                            mt,
                            nt,
                            kt,
                        ],
                    )
                ],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["out"]),
                [(ctx.core, [tensors["out"].buffer_address(), mt * nt])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                COMPUTE_KERNEL,
                ctx.grid,
                [mt, nt, kt, cfg["sub_mt"], cfg["sub_nt"]],
                [],
                compute,
            ),
        ]
        scale = ctx.torch["scale"].float()
        acc = ctx.torch["acc"].float()
        a = ctx.torch["a"].float()
        b = ctx.torch["b"].float()
        ref = scale * acc + a @ b
        return kernels, ref


if __name__ == "__main__":
    ScaledAccSweep().main()
