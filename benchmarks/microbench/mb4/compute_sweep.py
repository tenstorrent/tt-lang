# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB4 -- compute-op probe: per-tile compute-engine cost.

Applies a selected op to `tiles` tiles, `iters` times, and reports the per-RISC
zone times so the cost model has each op's tile cost. Ops, by category:

  unary SFPU  : copy (baseline), exp, gelu, recip, sqrt, rsqrt
  binary FPU  : add, mul                       (two full operands)
  broadcast   : mul_bcast, sub_bcast           (per-row scalar bcast over cols)
  reduce      : reduce_sum, reduce_max         (row reduction -> 1 tile)

The second operand is all ones, so values never affect timing and the PCC ref is
trivial (add->x+1, mul/mul_bcast->x, sub_bcast->x-1, reduce->rowsum/rowmax).
`op=copy` is the unary baseline: subtract it for the SFPU op's marginal math cost.
`--init-hoist 0` re-issues the op init every sub-block (init + op) instead of
hoisting it (steady op cost); reduce always hoists.

    python -m benchmarks.microbench.mb4.compute_sweep --op add,mul,mul_bcast,reduce_sum --tiles 1,2,4,8
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
UNARY_KERNEL = str(KERNELS / "compute" / "compute_unary.cpp")
BINARY_KERNEL = str(KERNELS / "compute" / "compute_binary.cpp")
REDUCE_KERNEL = str(KERNELS / "compute" / "compute_reduce.cpp")
READER_KERNEL = str(KERNELS / "compute" / "compute_op_reader.cpp")
WRITER_KERNEL = str(KERNELS / "common" / "drain_writer.cpp")

OP_IDS = {
    "copy": 0, "exp": 1, "gelu": 2, "recip": 3, "sqrt": 4, "rsqrt": 5,
    "add": 10, "mul": 11,
    "mul_bcast": 20, "sub_bcast": 21,
    "reduce_sum": 30, "reduce_max": 31,
}
BINARY_OPS = {"add", "mul"}
BCAST_OPS = {"mul_bcast", "sub_bcast"}
REDUCE_OPS = {"reduce_sum", "reduce_max"}
DEFAULT_OPS = ",".join(OP_IDS)


def _ref(op, x):
    """Reference for op(x), with the second operand fixed to ones."""
    if op == "copy" or op == "mul" or op == "mul_bcast":
        return x  # x*1 == x
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
    if op == "add":
        return x + 1.0
    if op == "sub_bcast":
        return x - 1.0  # x - ones[:, 0:1]
    if op in REDUCE_OPS:
        red = x.sum(dim=1) if op == "reduce_sum" else x.max(dim=1).values
        ref = torch.zeros(TILE, TILE, dtype=x.dtype)
        ref[:, 0] = red  # row reduction lands in column 0 of one tile
        return ref
    raise ValueError(f"unknown op {op!r}")


class ComputeOp(MicroBenchmark):
    NAME = "compute-op (unary / binary / bcast / reduce)"
    ZONE = "compute_op_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/compute_op.csv"
    PER_UNIT = "iters"
    CSV_TAG = ()  # dtype is swept (in the row), so it can't tag the filename
    EXTRA_COLUMNS = ("category", "out_tiles")
    PARAMS = (
        Param("op", DEFAULT_OPS, sweep=True, help="compute ops (see module doc)"),
        Param("tiles", "1,2,4,8", sweep=True, help="tiles per iteration"),
        Param("iters", "64", help="measured iterations"),
        Param("dtype", "bf16", sweep=True, help="bf16 and/or fp32"),
        Param(
            "init_hoist", "1", sweep=True, help="hoist op init out of the loop (0/1)"
        ),
        Param(
            "block_count", str(DEFAULT_BLOCK_COUNT), sweep=True, help="DFB block count"
        ),
    )
    # x positive so recip/sqrt/rsqrt are well-defined; y is ones (timing-neutral).
    INPUTS = (
        Tensor("x", lambda cfg: (TILE, cfg["tiles"] * TILE), scale=0.1, offset=1.0),
        Tensor(
            "y",
            lambda cfg: (TILE, (cfg["tiles"] if cfg["op"] in BINARY_OPS else 1) * TILE),
            init="ones",
        ),
    )
    OUTPUTS = (
        Tensor(
            "out",
            lambda cfg: (TILE, (1 if cfg["op"] in REDUCE_OPS else cfg["tiles"]) * TILE),
            init="empty",
        ),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["block_count"] * cfg["tiles"]),
        DFB(1, lambda cfg: cfg["block_count"] * (cfg["tiles"] if cfg["op"] in BINARY_OPS else 1)),
        DFB(16, lambda cfg: cfg["block_count"] * (1 if cfg["op"] in REDUCE_OPS else cfg["tiles"])),
    )

    def _n1(self, cfg):
        """Second-operand tile count the reader streams into cb1."""
        op = cfg["op"]
        if op in BINARY_OPS:
            return cfg["tiles"]
        if op in BCAST_OPS or op in REDUCE_OPS:
            return 1
        return 0  # unary: no second operand

    def _out_tiles(self, cfg):
        return 1 if cfg["op"] in REDUCE_OPS else cfg["tiles"]

    def _kernel(self, op):
        """Compute kernel for op's category (bcast shares the binary kernel)."""
        if op in REDUCE_OPS:
            return REDUCE_KERNEL
        if op in BINARY_OPS or op in BCAST_OPS:
            return BINARY_KERNEL
        return UNARY_KERNEL

    def min_pcc(self, cfg, strategy):
        if cfg["op"] == "gelu":
            return 0.98  # tanh approximation
        if cfg["op"] in REDUCE_OPS:
            return 0.95  # reduce result lives in column 0 only
        return 0.99

    def extra_columns(self, cfg, strategy):
        op = cfg["op"]
        category = (
            "reduce" if op in REDUCE_OPS
            else "bcast" if op in BCAST_OPS
            else "binary" if op in BINARY_OPS
            else "unary"
        )
        return {"category": category, "out_tiles": self._out_tiles(cfg)}

    def build(self, ctx):
        cfg = ctx.cfg
        tiles, iters = cfg["tiles"], cfg["iters"]
        fp32 = cfg["dtype"] == "fp32"
        cap = harness.dst_capacity(cfg["dtype"], False, fp32)
        init_hoist = int(cfg["init_hoist"])
        n1, out_tiles = self._n1(cfg), self._out_tiles(cfg)
        tensors = ctx.tensors
        compute = harness.compute_config(fp32_dest_acc=fp32)
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["x"], tensors["y"]),
                [
                    (
                        ctx.core,
                        [
                            tensors["x"].buffer_address(),
                            tensors["y"].buffer_address(),
                            tiles,
                            n1,
                        ],
                    )
                ],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["out"]),
                [(ctx.core, [tensors["out"].buffer_address(), out_tiles])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                self._kernel(cfg["op"]),
                ctx.grid,
                [OP_IDS[cfg["op"]], tiles, iters, cap, init_hoist],
                [],
                compute,
            ),
        ]
        ref = _ref(cfg["op"], ctx.torch["x"].float())
        return kernels, ref


if __name__ == "__main__":
    ComputeOp().main()
