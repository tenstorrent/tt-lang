# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Matmul diagnostic: per-node compute-feed utilization probe.

matmul_compute_sweep.py measures the matmul lowering cost model's compute-feed
term. It tests whether a handwritten generic-op matmul can feed the matrix
engine efficiently, using five diagnostic variants
(mm1_tile_loop .. mm5_block_stream_l1acc_packblock). The mm1->mm2 comparison
bundles the baseline-to-block-kernel changes; mm2->mm5 each change one
implementation detail. The resident variants (mm1, mm2) wait for operands
outside the timed zone; the streamed variants (mm3-mm5) wait on operand K-blocks
inside the zone, mirroring TTNN's large-block single-node compute contract.

    python -m benchmarks.microbench.mb3.matmul_compute_sweep --mt 4 --nt 4 --kt 8,16
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
COMPUTE_KERNEL = str(KERNELS / "matmul" / "matmul_tile_loop.cpp")
READER_KERNEL = str(KERNELS / "matmul" / "matmul_tile_loop_reader.cpp")
TTNN_LIKE_COMPUTE_KERNEL = str(KERNELS / "matmul" / "matmul_ttnn_like.cpp")
TTNN_LIKE_READER_KERNEL = str(KERNELS / "matmul" / "matmul_ttnn_like_reader.cpp")
TTNN_LIKE_READER_WRITER_KERNEL = str(KERNELS / "matmul" / "matmul_ttnn_like_reader_writer.cpp")
BLOCK_RESIDENT_COMPUTE_KERNEL = str(KERNELS / "matmul" / "matmul_block_resident.cpp")
L1ACC_COMPUTE_KERNEL = str(KERNELS / "matmul" / "matmul_block_stream_l1acc.cpp")
L1ACC_PACKBLOCK_COMPUTE_KERNEL = str(
    KERNELS / "matmul" / "matmul_block_stream_l1acc_packblock.cpp"
)
WRITER_KERNEL = str(KERNELS / "common" / "drain_writer.cpp")
MATMUL_CYCLES_PER_TILE = {"lofi": 16, "hifi2": 32, "hifi4": 64}
TTNN_SUBBLOCK_HW_CHOICES = (
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),
    (7, 1),
    (1, 7),
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),
    (5, 1),
    (1, 5),
    (2, 2),
    (4, 1),
    (1, 4),
    (3, 1),
    (1, 3),
    (2, 1),
    (1, 2),
    (1, 1),
)


class MatmulCompute(MicroBenchmark):
    NAME = "matmul compute-feed diagnostic"
    ZONE = "matmul_compute_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/matmul_compute.csv"
    STRATEGIES = (
        "mm1_tile_loop",
        "mm2_block",
        "mm3_block_stream",
        "mm4_block_stream_l1acc",
        "mm5_block_stream_l1acc_packblock",
    )
    PER_UNIT = "kt"
    CSV_TAG = ("dtype", "fidelity", "full_sync")
    EXTRA_COLUMNS = ("sub_mt", "sub_nt", "reuse", "in0_block_w")
    POST_COLUMNS = (
        "matmul_ideal_cycles",
        "trisc_max_cycles",
        "math_cycles",
        "zone_utilization_pct",
        "math_utilization_pct",
    )
    PARAMS = (
        Param("mt", "4", sweep=True, help="output rows (tiles)"),
        Param("nt", "4", sweep=True, help="output cols (tiles)"),
        Param("kt", "8,16,32", sweep=True, help="K-depth (tiles)"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("fidelity", "hifi4", choices=("lofi", "hifi2", "hifi4")),
        Param("full_sync", False),
        Param(
            "in0_block_w_div",
            "1",
            help="TTNN-like K-block divisor: in0_block_w = kt / in0_block_w_div",
        ),
        Param(
            "block_count",
            "1",
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
    DFBS = (
        DFB(0, lambda cfg, strategy: MatmulCompute._operand_pages(cfg, strategy, "a")),
        DFB(1, lambda cfg, strategy: MatmulCompute._operand_pages(cfg, strategy, "b")),
        DFB(4, lambda cfg: cfg["mt"] * cfg["nt"]),
        DFB(5, lambda cfg: cfg["mt"] * cfg["nt"]),
        DFB(16, lambda cfg: cfg["block_count"] * cfg["mt"] * cfg["nt"]),
    )

    @staticmethod
    def _operand_pages(cfg, strategy, operand):
        # Resident variants (mm1, mm2) hold the whole operand in L1. Streamed
        # variants (mm3-mm5) only need a double-buffered K block, so they stay
        # within L1 (Wormhole could not fit the whole-operand sizing) and match
        # TTNN's block dataflow buffers. block_count sets prefetch depth, floored
        # at 2 so the next block's load overlaps the current block's compute.
        in0_block_w = MatmulCompute._in0_block_w(cfg)
        num_blocks = cfg["kt"] // in0_block_w
        resident = strategy in ("mm1_tile_loop", "mm2_block")
        depth = min(num_blocks, max(2, cfg["block_count"]))
        if operand == "a":
            if resident:
                return cfg["mt"] * cfg["kt"]
            return depth * cfg["mt"] * in0_block_w
        if resident:
            return cfg["kt"] * cfg["nt"]
        return depth * in0_block_w * cfg["nt"]

    @staticmethod
    def _in0_block_w(cfg):
        divisor = cfg["in0_block_w_div"]
        if divisor <= 0:
            raise ValueError("in0_block_w_div must be positive")
        if cfg["kt"] % divisor != 0:
            raise ValueError("kt must be divisible by in0_block_w_div")
        return cfg["kt"] // divisor

    @staticmethod
    def _compiler_subblock(cfg):
        cap = harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )
        return harness.dst_subblock(cfg["mt"], cfg["nt"], cap)

    @staticmethod
    def _ttnn_subblock(cfg):
        fp32_dest_acc = cfg["dtype"] == "fp32"
        for sub_mt, sub_nt in TTNN_SUBBLOCK_HW_CHOICES:
            if fp32_dest_acc and sub_mt * sub_nt > 4:
                continue
            if cfg["mt"] % sub_mt == 0 and cfg["nt"] % sub_nt == 0:
                return sub_mt, sub_nt
        return 1, 1

    def legal(self, cfg, strategy):
        if cfg["in0_block_w_div"] <= 0:
            return False
        if cfg["kt"] % cfg["in0_block_w_div"] != 0:
            return False
        # mm2_block keeps the whole K block resident (num_blocks == 1), so it
        # isolates matmul+pack from operand load without K streaming.
        if strategy == "mm2_block" and cfg["in0_block_w_div"] != 1:
            return False
        return True

    def _subblock(self, cfg, strategy):
        if strategy == "mm1_tile_loop":
            return self._compiler_subblock(cfg)
        return self._ttnn_subblock(cfg)

    def extra_columns(self, cfg, strategy):
        sub_mt, sub_nt = self._subblock(cfg, strategy)
        reuse = (cfg["mt"] // sub_mt) * (cfg["nt"] // sub_nt)
        return {
            "sub_mt": sub_mt,
            "sub_nt": sub_nt,
            "reuse": reuse,
            "in0_block_w": self._in0_block_w(cfg),
        }

    def post_columns(self, cfg, strategy, zone_summary, row):
        tile_matmul_count = cfg["mt"] * cfg["nt"] * cfg["kt"]
        matmul_ideal_cycles = (
            tile_matmul_count * MATMUL_CYCLES_PER_TILE[cfg["fidelity"]]
        )
        freq_mhz = zone_summary["freq_mhz"]

        def cycles(microseconds):
            if microseconds is None:
                return None
            return round(microseconds * freq_mhz, 2)

        def utilization_pct(actual_cycles):
            if actual_cycles is None or actual_cycles <= 0:
                return None
            return round(100.0 * matmul_ideal_cycles / actual_cycles, 2)

        trisc_max_cycles = cycles(zone_summary["trisc_max_us"])
        math_cycles = cycles(zone_summary["math_us"])
        return {
            "matmul_ideal_cycles": matmul_ideal_cycles,
            "trisc_max_cycles": trisc_max_cycles,
            "math_cycles": math_cycles,
            "zone_utilization_pct": utilization_pct(trisc_max_cycles),
            "math_utilization_pct": utilization_pct(math_cycles),
        }

    def build(self, ctx):
        cfg = ctx.cfg
        mt, nt, kt = cfg["mt"], cfg["nt"], cfg["kt"]
        sub_mt, sub_nt = self._subblock(cfg, ctx.strategy)
        in0_block_w = self._in0_block_w(cfg)
        tensors = ctx.tensors
        compute = harness.compute_config(
            cfg["fidelity"],
            fp32_dest_acc=cfg["dtype"] == "fp32",
            full_sync=cfg["full_sync"],
        )
        streamed_kernels = {
            "mm3_block_stream": TTNN_LIKE_COMPUTE_KERNEL,
            "mm4_block_stream_l1acc": L1ACC_COMPUTE_KERNEL,
            "mm5_block_stream_l1acc_packblock": L1ACC_PACKBLOCK_COMPUTE_KERNEL,
        }
        if ctx.strategy in streamed_kernels:
            compute.math_approx_mode = True
            return self._build_streamed(
                ctx,
                tensors,
                compute,
                mt,
                nt,
                kt,
                in0_block_w,
                sub_mt,
                sub_nt,
                streamed_kernels[ctx.strategy],
            )
        if ctx.strategy == "mm2_block":
            compute.math_approx_mode = True
            return self._build_block_resident(
                ctx, tensors, compute, mt, nt, kt, sub_mt, sub_nt
            )

        reader_runtime_args = [
            tensors["a"].buffer_address(),
            tensors["b"].buffer_address(),
            mt,
            nt,
            kt,
        ]
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["a"], tensors["b"]),
                [(ctx.core, reader_runtime_args)],
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
                COMPUTE_KERNEL,
                ctx.grid,
                [mt, nt, kt, sub_mt, sub_nt],
                [],
                compute,
            ),
        ]
        ref = ctx.torch["a"].float() @ ctx.torch["b"].float()
        return kernels, ref

    def _build_streamed(
        self,
        ctx,
        tensors,
        compute,
        mt,
        nt,
        kt,
        in0_block_w,
        sub_mt,
        sub_nt,
        compute_kernel,
    ):
        kernels = [
            harness.file_kernel(
                TTNN_LIKE_READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["a"]),
                [
                    (
                        ctx.core,
                        [
                            tensors["a"].buffer_address(),
                            mt,
                            kt,
                            in0_block_w,
                        ],
                    )
                ],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                TTNN_LIKE_READER_WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["b"], tensors["c"]),
                [
                    (
                        ctx.core,
                        [
                            tensors["b"].buffer_address(),
                            tensors["c"].buffer_address(),
                            mt,
                            nt,
                            kt,
                            in0_block_w,
                            sub_mt,
                            sub_nt,
                        ],
                    )
                ],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                compute_kernel,
                ctx.grid,
                [mt, nt, kt, in0_block_w, sub_mt, sub_nt],
                [],
                compute,
            ),
        ]
        ref = ctx.torch["a"].float() @ ctx.torch["b"].float()
        return kernels, ref

    def _build_block_resident(self, ctx, tensors, compute, mt, nt, kt, sub_mt, sub_nt):
        # Reuse the ttnn_like readers with in0_block_w = kt (one K block, whole
        # operand resident); the compute kernel waits outside the timed zone.
        kernels = [
            harness.file_kernel(
                TTNN_LIKE_READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["a"]),
                [(ctx.core, [tensors["a"].buffer_address(), mt, kt, kt])],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                TTNN_LIKE_READER_WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["b"], tensors["c"]),
                [
                    (
                        ctx.core,
                        [
                            tensors["b"].buffer_address(),
                            tensors["c"].buffer_address(),
                            mt,
                            nt,
                            kt,
                            kt,
                            sub_mt,
                            sub_nt,
                        ],
                    )
                ],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                BLOCK_RESIDENT_COMPUTE_KERNEL,
                ctx.grid,
                [mt, nt, kt, sub_mt, sub_nt],
                [],
                compute,
            ),
        ]
        ref = ctx.torch["a"].float() @ ctx.torch["b"].float()
        return kernels, ref


if __name__ == "__main__":
    MatmulCompute().main()
