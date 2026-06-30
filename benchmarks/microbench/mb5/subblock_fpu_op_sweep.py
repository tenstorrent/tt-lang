# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""FPU-op subblock sweep (MB5.B): how the subblock shape affects a reuse-free FPU
op, across ops.

A single FPU op is applied per tile over a square `n x n` block, subblocked by a
configurable `(sub_h, sub_w)` chunk, repeated over a resident input (CB hop
outside the zone). Ops (all FPU, all reading the input straight into srcA/srcB --
no copy_tile):

  transpose : transpose_wh per tile, packed to the transposed grid position
              (access pattern depends on subblock shape)
  add       : add_tiles(x, x) = 2x   (binary FPU, same position)
  mul       : mul_tiles(x, x) = x^2  (binary FPU, same position)

Binary ops feed the input as both operands, so there is no second operand buffer
(footprint = 2*n^2 tiles for every op). The `compiler_pick` column flags the
maximize-product subblock; comparing dst_chunk and the per-RISC split across ops
shows whether the thin-vs-blocky shape effect is transpose-specific.

    python -m benchmarks.microbench.mb5.subblock_fpu_op_sweep --op transpose,add,mul --n 4,8,12,16
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

from pathlib import Path

import ttnn

from benchmarks.microbench import harness
from benchmarks.microbench.harness import TILE
from benchmarks.microbench.runner import DFB, MicroBenchmark, Param, Tensor

KERNELS = Path("benchmarks/microbench/kernels")
COMPUTE_KERNEL = str(KERNELS / "compute" / "fpu_op_compute.cpp")
READER_KERNEL = str(KERNELS / "common" / "seed_reader.cpp")
WRITER_KERNEL = str(KERNELS / "common" / "drain_writer.cpp")

OP_IDS = {"transpose": 0, "add": 1, "mul": 2}
# Footprint is cb_in (n^2) + cb_out (n^2) = 2*n^2 tiles; keep under the L1 budget.
MAX_FOOTPRINT_TILES = 700


def _ref(op, x):
    if op == "transpose":
        return x.t()
    if op == "add":
        return x + x
    if op == "mul":
        return x * x
    raise ValueError(f"unknown op {op!r}")


class FpuOpSweep(MicroBenchmark):
    NAME = "FPU op subblock sweep"
    ZONE = "fpu_op_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/subblock_fpu_op.csv"
    STRATEGIES = ("",)
    PER_UNIT = "iters"
    CSV_TAG = ("dtype", "full_sync")
    EXTRA_COLUMNS = ("dst_chunk", "acquires", "compiler_pick")
    PARAMS = (
        Param("op", "transpose,add,mul", sweep=True, help="FPU op"),
        Param("n", "4,8,12,16", sweep=True, help="square block side (tiles)"),
        Param("iters", "128", help="measured iterations"),
        Param("sub_h", "1,2,4,8", sweep=True, help="subblock rows"),
        Param("sub_w", "1,2,4,8", sweep=True, help="subblock cols"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("full_sync", False),
    )
    INPUTS = (
        Tensor("x", lambda cfg: (cfg["n"] * TILE, cfg["n"] * TILE), scale=0.1, offset=1.0),
    )
    OUTPUTS = (
        Tensor("out", lambda cfg: (cfg["n"] * TILE, cfg["n"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["n"] * cfg["n"]),  # input (resident)
        DFB(16, lambda cfg: cfg["n"] * cfg["n"]),  # output (drained once)
    )

    def _cap(self, cfg):
        return harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )

    def legal(self, cfg, strategy):
        n, sh, sw = cfg["n"], cfg["sub_h"], cfg["sub_w"]
        if n % sh or n % sw:
            return False
        if sh * sw > self._cap(cfg):
            return False
        return 2 * n * n <= MAX_FOOTPRINT_TILES  # fits L1

    def extra_columns(self, cfg, strategy):
        n, sh, sw = cfg["n"], cfg["sub_h"], cfg["sub_w"]
        pick = harness.dst_subblock(n, n, self._cap(cfg))
        return {
            "dst_chunk": sh * sw,
            "acquires": (n // sh) * (n // sw),
            "compiler_pick": int((sh, sw) == pick),
        }

    def summary(self, cfg, by_strategy):
        row = next(iter(by_strategy.values()))
        per_iter = row["trisc_max_us_per_iters"]
        per_iter = "n/a" if per_iter is None else f"{per_iter:.4f}"
        star = "  <== compiler pick" if row["compiler_pick"] else ""
        return (
            f"op={cfg['op']:<9} n={cfg['n']:>2} sub=({cfg['sub_h']},{cfg['sub_w']}) "
            f"chunk={row['dst_chunk']} acq={row['acquires']:>3} "
            f"| per_iter={per_iter} µs pcc={row['pcc']}{star}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        n, iters = cfg["n"], cfg["iters"]
        tensors = ctx.tensors
        compute = harness.compute_config(
            fp32_dest_acc=cfg["dtype"] == "fp32", full_sync=cfg["full_sync"]
        )
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["x"]),
                [(ctx.core, [tensors["x"].buffer_address(), n * n])],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["out"]),
                [(ctx.core, [tensors["out"].buffer_address(), n * n])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                COMPUTE_KERNEL,
                ctx.grid,
                [n, n, iters, cfg["sub_h"], cfg["sub_w"], OP_IDS[cfg["op"]]],
                [],
                compute,
            ),
        ]
        ref = _ref(cfg["op"], ctx.torch["x"].float())
        return kernels, ref


if __name__ == "__main__":
    FpuOpSweep().main()
