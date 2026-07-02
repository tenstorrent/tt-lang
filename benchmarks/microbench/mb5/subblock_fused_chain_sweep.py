# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fused-chain subblock sweep (MB5 variant): DST chunk vs total tiles for a real
tt-lang-generated fused SFPU chain.

Drives fused_chain_compute.cpp -- the body extracted verbatim from the tt-lang
codegen for  v = abs(neg(relu(sigmoid(c) + tanh(b)))) -- over a flat array of
`tiles` output tiles, subblocked by `sub` = output tiles per tile_regs_acquire.
Same flat, resident, hop-outside style as the compute_unary/copy sweeps.

The fused add keeps both operands live, so each output tile uses TWO dst slots;
the subblock is bounded by 2*sub <= DST capacity (sub in {1,2,4} at half-sync
cap 8, up to {..,8} at full-sync cap 16). The `dst_budget` column reports 2*sub.

    python -m benchmarks.microbench.mb5.subblock_fused_chain_sweep \
        --tiles 8,16,32,64,128 --sub 1,2,4
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
COMPUTE_KERNEL = str(KERNELS / "compute" / "fused_chain_compute.cpp")
READER_KERNEL = str(KERNELS / "compute" / "compute_op_reader.cpp")
WRITER_KERNEL = str(KERNELS / "common" / "drain_writer.cpp")


def _ref(b, c):
    v = torch.tanh(b)
    v = torch.sigmoid(c) + v
    v = torch.relu(v)
    v = torch.neg(v)
    v = torch.abs(v)
    return v


class FusedChainSweep(MicroBenchmark):
    NAME = "fused-chain (sigmoid(c)+tanh(b) -> relu -> neg -> abs) subblock sweep"
    ZONE = "fused_chain_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/subblock_fused_chain.csv"
    STRATEGIES = ("",)
    CSV_TAG = ("dtype", "full_sync")
    EXTRA_COLUMNS = ("dst_chunk", "dst_budget", "acquires", "max_dst_pick")
    PARAMS = (
        Param("tiles", "8,16,32,64,128", sweep=True, help="total output tiles"),
        Param("sub", "1,2,4,8", sweep=True, help="output tiles per acquire"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("full_sync", "0", sweep=True, help="dst_full_sync_en (0/1)"),
        Param("fp32_dest_acc", "0", sweep=True, help="fp32_dest_acc_en (0/1)"),
    )
    # b positive-ish and c arbitrary; scale keeps values in a well-behaved range
    # for tanh/sigmoid so bf16 PCC stays high.
    INPUTS = (
        Tensor("b", lambda cfg: (TILE, cfg["tiles"] * TILE), scale=0.5),
        Tensor("c", lambda cfg: (TILE, cfg["tiles"] * TILE), scale=0.5),
    )
    OUTPUTS = (
        Tensor("out", lambda cfg: (TILE, cfg["tiles"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["tiles"]),   # b (resident)
        DFB(1, lambda cfg: cfg["tiles"]),   # c (resident)
        DFB(16, lambda cfg: cfg["tiles"]),  # out (drained once)
    )
    # SFPU chain (tanh/sigmoid approximations) -- match MB2/MB3 gelu tolerance.
    MIN_PCC = 0.98

    def _cap(self, cfg):
        full_sync, fp32 = bool(int(cfg["full_sync"])), bool(int(cfg["fp32_dest_acc"]))
        return harness.dst_capacity(cfg["dtype"], full_sync, fp32)

    def legal(self, cfg, strategy):
        sub = cfg["sub"]
        if sub <= 0 or sub > cfg["tiles"]:
            return False
        # each output tile uses two dst slots (fused add keeps both operands live)
        return 2 * sub <= self._cap(cfg)

    def extra_columns(self, cfg, strategy):
        sub, cap = cfg["sub"], self._cap(cfg)
        return {
            "dst_chunk": sub,
            "dst_budget": 2 * sub,
            "acquires": -(-cfg["tiles"] // sub),
            # max subblock under the 2-slots-per-tile budget, capped by tiles
            "max_dst_pick": int(sub == min(cap // 2, cfg["tiles"])),
        }

    def summary(self, cfg, by_strategy):
        row = next(iter(by_strategy.values()))
        star = "  <== max-DST" if row["max_dst_pick"] else ""
        return (
            f"tiles={cfg['tiles']:>3} sub={cfg['sub']} dst={row['dst_budget']:>2} "
            f"acq={row['acquires']:>3} fs={cfg['full_sync']} | "
            f"unpack={row['unpack_us']} math={row['math_us']} pack={row['pack_us']} "
            f"trisc_max={row['trisc_max_us']} µs pcc={row['pcc']}{star}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        tiles, sub = cfg["tiles"], cfg["sub"]
        full_sync, fp32 = bool(int(cfg["full_sync"])), bool(int(cfg["fp32_dest_acc"]))
        tensors = ctx.tensors
        compute = harness.compute_config(fp32_dest_acc=fp32, full_sync=full_sync)
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["b"], tensors["c"]),
                [
                    (
                        ctx.core,
                        # compute_op_reader: 0=x(->cb0) addr, 1=y(->cb1) addr,
                        # 2=n0 (cb0 tiles), 3=n1 (cb1 tiles). cb0=b, cb1=c.
                        [
                            tensors["b"].buffer_address(),
                            tensors["c"].buffer_address(),
                            tiles,
                            tiles,
                        ],
                    )
                ],
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
                [tiles, sub],
                [],
                compute,
            ),
        ]
        ref = _ref(ctx.torch["b"].float(), ctx.torch["c"].float())
        return kernels, ref


if __name__ == "__main__":
    FusedChainSweep().main()
