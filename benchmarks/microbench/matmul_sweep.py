# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB3 -- matmul K-accumulation: DST-K vs L1-K (production-representative).

C[mt,nt] = sum_k A[k] @ B[k] over kt K-tiles, two strategies for where the
running partial lives. The output is tiled into sub_mt*sub_nt subblocks chosen
exactly as the compiler would (harness.dst_subblock), giving the production
reuse factor.

  MB3.A (mt*nt <= DST capacity): one subblock. DST-K holds the whole output in
    DST across the K loop and packs once; L1-K repacks it to L1 every K step.
    Operands are unpacked once.
  MB3.B (mt*nt > DST capacity): reuse > 1 subblocks. DST-K re-unpacks each
    subblock's operands from L1 across the K loop (unpack-dominated); L1-K still
    repacks every subblock every K step (pack-dominated). The regime where the
    per-engine weights determine the lower-cost strategy.

--fuse gelu adds a GELU epilogue: DST-K applies it in place on the resident
output before its single pack; L1-K must reload the L1 accumulator into DST,
apply, and repack -- the round trip that can favor DST-K when an epilogue is
present.

    python -m benchmarks.microbench.matmul_sweep --mt 1,2 --nt 1,2 --kt 1,2,4,8,16
    python -m benchmarks.microbench.matmul_sweep --mt 4 --nt 4 --kt 4,8 --fuse gelu
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
    CSV_TAG = ("dtype", "fidelity", "fuse", "full_sync")
    EXTRA_COLUMNS = ("sub_mt", "sub_nt", "reuse")
    PARAMS = (
        Param("mt", "1", sweep=True, help="output rows (tiles)"),
        Param("nt", "1", sweep=True, help="output cols (tiles)"),
        Param("kt", "1,2,4,8,16", sweep=True, help="K-depth (tiles)"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("fidelity", "hifi4", choices=("lofi", "hifi2", "hifi4")),
        Param("full_sync", False),
        Param("fuse", "none", choices=("none", "gelu"), help="epilogue activation"),
        Param(
            "block_count",
            str(DEFAULT_BLOCK_COUNT),
            sweep=True,
            help="output DFB block count",
        ),
    )
    INPUTS = (
        Tensor("a", lambda cfg: (cfg["mt"] * TILE, cfg["kt"] * TILE)),
        Tensor("b", lambda cfg: (cfg["kt"] * TILE, cfg["nt"] * TILE)),
    )
    OUTPUTS = (
        Tensor("c", lambda cfg: (cfg["mt"] * TILE, cfg["nt"] * TILE), init="empty"),
    )
    # DST-K keeps all operands resident to re-unpack them per output subblock;
    # L1-K streams per K step but tolerates the larger DFBs. dfb_acc (index 2)
    # is the L1-K fused accumulator, reloaded for the epilogue.
    DFBS = (
        DFB(0, lambda cfg: cfg["kt"] * cfg["mt"]),
        DFB(1, lambda cfg: cfg["kt"] * cfg["nt"]),
        DFB(2, lambda cfg: cfg["mt"] * cfg["nt"]),
        DFB(16, lambda cfg: cfg["block_count"] * cfg["mt"] * cfg["nt"]),
    )

    def min_pcc(self, cfg, strategy):
        # The fused epilogue uses fast GELU (tanh approximation) -- the production
        # default for matmul fusion, and far cheaper than the erf-precise variant,
        # which would dominate the kernel and hide the strategy comparison. Its
        # approximation lowers PCC to ~0.985; tt-metal/tt-blaze accept this for
        # fused matmul+activation. Plain matmul stays at the tight default.
        return 0.98 if cfg["fuse"] == "gelu" else self.MIN_PCC

    def _subblock(self, cfg):
        cap = harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )
        return harness.dst_subblock(cfg["mt"], cfg["nt"], cap)

    def extra_columns(self, cfg, strategy):
        sub_mt, sub_nt = self._subblock(cfg)
        reuse = (cfg["mt"] // sub_mt) * (cfg["nt"] // sub_nt)
        return {"sub_mt": sub_mt, "sub_nt": sub_nt, "reuse": reuse}

    def build(self, ctx):
        cfg = ctx.cfg
        mt, nt, kt = cfg["mt"], cfg["nt"], cfg["kt"]
        sub_mt, sub_nt = self._subblock(cfg)
        fuse = 1 if cfg["fuse"] == "gelu" else 0
        tensors = ctx.tensors
        compute = harness.compute_config(
            cfg["fidelity"],
            fp32_dest_acc=cfg["dtype"] == "fp32",
            full_sync=cfg["full_sync"],
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
                [mt, nt, kt, sub_mt, sub_nt, fuse],
                [],
                compute,
            ),
        ]
        ref = ctx.torch["a"].float() @ ctx.torch["b"].float()
        if cfg["fuse"] == "gelu":
            # Fast SFPU GELU is the tanh approximation; match the reference.
            ref = torch.nn.functional.gelu(ref, approximate="tanh")
        return kernels, ref


if __name__ == "__main__":
    MatmulK().main()
