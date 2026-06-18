# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB4 -- compute-op (math) probe: per-tile SFPU math-engine cost.

Applies a selected SFPU unary op to `tiles` tiles, `iters` times, on the math
thread (what tt-lang emits), and reports the per-RISC zone times. op=copy is the
baseline (copy into DST, no SFPU); the marginal op cost is the math-thread delta
over copy. `--init-hoist 0` re-issues the op init every sub-block (init + op
cost) instead of hoisting it (steady op cost). See runner.py.

    python -m benchmarks.microbench.compute_sweep --op copy,exp,gelu,recip,sqrt,rsqrt --tiles 1,2,4,8
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
COMPUTE_KERNEL = str(KERNELS / "compute_unary.cpp")
READER_KERNEL = str(KERNELS / "seed_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")

OP_IDS = {"copy": 0, "exp": 1, "gelu": 2, "recip": 3, "sqrt": 4, "rsqrt": 5}
OP_REF = {
    "copy": lambda x: x,
    "exp": torch.exp,
    "gelu": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
    "recip": torch.reciprocal,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
}


class ComputeOp(MicroBenchmark):
    NAME = "compute-op (SFPU unary)"
    ZONE = "compute_op_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/compute_op.csv"
    PER_UNIT = "iters"
    CSV_TAG = ()  # dtype is swept (in the row), so it can't tag the filename
    PARAMS = (
        Param(
            "op", "copy,exp,gelu,recip,sqrt,rsqrt", sweep=True, help="SFPU unary ops"
        ),
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
    # Positive input keeps recip/sqrt/rsqrt well-defined.
    INPUTS = (
        Tensor("x", lambda cfg: (TILE, cfg["tiles"] * TILE), scale=0.1, offset=1.0),
    )
    OUTPUTS = (Tensor("out", lambda cfg: (TILE, cfg["tiles"] * TILE), init="empty"),)
    DFBS = (
        DFB(0, lambda cfg: cfg["block_count"] * cfg["tiles"]),
        DFB(16, lambda cfg: cfg["block_count"] * cfg["tiles"]),
    )

    def min_pcc(self, cfg, strategy):
        # Fast (approximate) SFPU ops; gelu's tanh approximation is the loosest.
        return 0.98 if cfg["op"] == "gelu" else 0.99

    def build(self, ctx):
        cfg = ctx.cfg
        tiles, iters = cfg["tiles"], cfg["iters"]
        fp32 = cfg["dtype"] == "fp32"
        cap = harness.dst_capacity(cfg["dtype"], False, fp32)
        init_hoist = int(cfg["init_hoist"])
        tensors = ctx.tensors
        compute = harness.compute_config(fp32_dest_acc=fp32)
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
                [OP_IDS[cfg["op"]], tiles, iters, cap, init_hoist],
                [],
                compute,
            ),
        ]
        ref = OP_REF[cfg["op"]](ctx.torch["x"].float())
        return kernels, ref


if __name__ == "__main__":
    ComputeOp().main()
