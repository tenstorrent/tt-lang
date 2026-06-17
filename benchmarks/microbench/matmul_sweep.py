# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB3 — matmul K-accumulation: DST-K vs L1-K (production-representative).

C[mt,nt] = sum_k A[k] @ B[k] over kt K-tiles. DST-K holds the mt*nt output
subblock in DST across the K loop (matmul_block, pack once); L1-K packs the
subblock to L1 each K step (pack_reconfig_l1_acc). Declared on the MicroBenchmark
runner (see runner.py); DST-K is legal only while mt*nt <= getDstCapacity.

    python -m benchmarks.microbench.matmul_sweep --mt 1,2 --nt 1,2 --kt 1,2,4,8,16
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")

from pathlib import Path

import ttnn

from benchmarks.microbench import harness
from benchmarks.microbench.harness import BLOCK_COUNT, TILE
from benchmarks.microbench.runner import DFB, MicroBenchmark, Param, Tensor

KERNELS = Path("benchmarks/microbench/kernels")
DST_KERNEL = str(KERNELS / "matmul_k_dst.cpp")
L1_KERNEL = str(KERNELS / "matmul_k_l1.cpp")
READER_KERNEL = str(KERNELS / "matmul_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")


class MatmulK(MicroBenchmark):
    NAME = "matmul DST-K vs L1-K"
    ZONE = "matmul_k_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/matmul_k.csv"
    STRATEGIES = ("dst", "l1")
    PER_UNIT = "kt"
    CSV_TAG = ("dtype", "fidelity")
    PARAMS = (
        Param("mt", "1", sweep=True, help="output rows (tiles)"),
        Param("nt", "1", sweep=True, help="output cols (tiles)"),
        Param("kt", "1,2,4,8,16", sweep=True, help="K-depth (tiles)"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("fidelity", "hifi4", choices=("lofi", "hifi2", "hifi4")),
    )
    INPUTS = (
        Tensor("a", lambda cfg: (cfg["mt"] * TILE, cfg["kt"] * TILE)),
        Tensor("b", lambda cfg: (cfg["kt"] * TILE, cfg["nt"] * TILE)),
    )
    OUTPUTS = (
        Tensor("c", lambda cfg: (cfg["mt"] * TILE, cfg["nt"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: BLOCK_COUNT * cfg["mt"]),
        DFB(1, lambda cfg: BLOCK_COUNT * cfg["nt"]),
        DFB(16, lambda cfg: BLOCK_COUNT * cfg["mt"] * cfg["nt"]),
    )
    CSV_COLUMNS = (
        "arch",
        "dtype",
        "fidelity",
        "strategy",
        "mt",
        "nt",
        "kt",
        "freq_mhz",
        "trisc_max_us",
        "trisc_max_us_per_kt",
        "unpack_us",
        "math_us",
        "pack_us",
        "noc_active_in_zone",
        "pcc",
    )

    def build(self, ctx):
        cfg = ctx.cfg
        mt, nt, kt = cfg["mt"], cfg["nt"], cfg["kt"]
        tensors = ctx.tensors
        compute = harness.compute_config(
            cfg["fidelity"], fp32_dest_acc=cfg["dtype"] == "fp32"
        )
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["a"], tensors["b"]),
                [
                    (
                        ctx.core,
                        [
                            tensors["a"].buffer_address(),
                            tensors["b"].buffer_address(),
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
                harness.accessor_args(tensors["c"]),
                [(ctx.core, [tensors["c"].buffer_address(), mt * nt])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                DST_KERNEL if ctx.strategy == "dst" else L1_KERNEL,
                ctx.grid,
                [mt, nt, kt],
                [],
                compute,
            ),
        ]
        ref = ctx.torch["a"].float() @ ctx.torch["b"].float()
        return kernels, ref


if __name__ == "__main__":
    MatmulK().main()
