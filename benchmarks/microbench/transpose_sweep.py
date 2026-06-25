# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""FPU transpose subblock sweep -- Y = transpose(X).

A standalone block-transpose (transpose_wh per tile + transposed grid placement),
the way tt-lang subblocks each ttl.compute region for DST. The transpose is a real
FPU op with strictly 1:1 tile mapping (zero operand reuse), so the subblock knob
is purely the per-acquire DST chunk -- the same lever as MB1's pack/unpack probe,
now with FPU math in the loop. Holds the (R x C) tile grid fixed and force-sweeps
the output subblock (sub_h, sub_w); `compiler_pick` flags the maximize-product
choice the compiler would make, so the sweep shows whether that is actually best.

    python -m benchmarks.microbench.transpose_sweep --r 8 --c 8
    python -m benchmarks.microbench.transpose_sweep --r 8 --c 8 --full-sync
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
COMPUTE_KERNEL = str(KERNELS / "transpose_compute.cpp")
READER_KERNEL = str(KERNELS / "seed_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")


class TransposeSweep(MicroBenchmark):
    NAME = "FPU transpose subblock sweep"
    ZONE = "transpose_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/transpose.csv"
    STRATEGIES = ("",)
    PER_UNIT = "iters"
    CSV_TAG = ("dtype", "full_sync")
    EXTRA_COLUMNS = ("dst_chunk", "acquires", "compiler_pick")
    PARAMS = (
        Param("r", "8", help="input rows (tiles)"),
        Param("c", "8", help="input cols (tiles)"),
        Param("iters", "128", help="measured iterations"),
        Param("sub_h", "1,2,4,8", sweep=True, help="subblock rows"),
        Param("sub_w", "1,2,4,8", sweep=True, help="subblock cols"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("full_sync", False),
        Param("block_count", str(DEFAULT_BLOCK_COUNT), sweep=True),
    )
    INPUTS = (Tensor("x", lambda cfg: (cfg["r"] * TILE, cfg["c"] * TILE), scale=0.1),)
    OUTPUTS = (
        Tensor("y", lambda cfg: (cfg["c"] * TILE, cfg["r"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["r"] * cfg["c"]),  # x (resident)
        DFB(16, lambda cfg: cfg["block_count"] * cfg["r"] * cfg["c"]),  # y
    )

    def _cap(self, cfg):
        return harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )

    def legal(self, cfg, strategy):
        sh, sw = cfg["sub_h"], cfg["sub_w"]
        if cfg["r"] % sh or cfg["c"] % sw:
            return False
        return sh * sw <= self._cap(cfg)

    def extra_columns(self, cfg, strategy):
        sh, sw = cfg["sub_h"], cfg["sub_w"]
        acquires = (cfg["r"] // sh) * (cfg["c"] // sw)
        pick = harness.dst_subblock(cfg["r"], cfg["c"], self._cap(cfg))
        return {
            "dst_chunk": sh * sw,
            "acquires": acquires,
            "compiler_pick": int((sh, sw) == pick),
        }

    def summary(self, cfg, by_strategy):
        row = next(iter(by_strategy.values()))
        per_iter = row["trisc_max_us_per_iters"]
        per_iter = "n/a" if per_iter is None else f"{per_iter:.4f}"
        star = "  <== compiler-pick" if row["compiler_pick"] else ""
        return (
            f"r={cfg['r']} c={cfg['c']} sub=({cfg['sub_h']},{cfg['sub_w']}) "
            f"chunk={row['dst_chunk']} acquires={row['acquires']} "
            f"| trisc_max={row['trisc_max_us']} µs per_iter={per_iter} µs "
            f"math={row['math_us']} pack={row['pack_us']} pcc={row['pcc']}{star}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        r, c = cfg["r"], cfg["c"]
        tensors = ctx.tensors
        compute = harness.compute_config(
            fp32_dest_acc=cfg["dtype"] == "fp32", full_sync=cfg["full_sync"]
        )
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["x"]),
                [(ctx.core, [tensors["x"].buffer_address(), r * c])],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["y"]),
                [(ctx.core, [tensors["y"].buffer_address(), r * c])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                COMPUTE_KERNEL,
                ctx.grid,
                [r, c, cfg["iters"], cfg["sub_h"], cfg["sub_w"]],
                [],
                compute,
            ),
        ]
        ref = ctx.torch["x"].float().t()  # Y = transpose(X)
        return kernels, ref


if __name__ == "__main__":
    TransposeSweep().main()
