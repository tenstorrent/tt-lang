# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-pass (non-flash) SDPA, every compute region subblocked by one
configurable (sub_h, sub_w) -- the way tt-lang subblocks each ttl.compute for
DST. Sweeping (sub_h, sub_w) drives the whole kernel's tiling (QK^T, exp,
rowsum, recip, bcast, P@V, normalize), so it measures the end-to-end effect of
the subblock-size choice the compiler makes per region.

Sk = HD by default so one sub_w divides both the score (Sq x Sk) and output
(Sq x HD) regions. Correctness is PCC against torch softmax(Q @ Kt) @ V.

    python -m benchmarks.microbench.sdpa_sweep
    python -m benchmarks.microbench.sdpa_sweep --sq 8 --sk 8 --hd 8 --fidelity lofi
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
COMPUTE_KERNEL = str(KERNELS / "sdpa_compute.cpp")
READER_KERNEL = str(KERNELS / "sdpa_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")
MATMUL_CYCLES_PER_TILE = {"lofi": 16, "hifi2": 32, "hifi4": 64}


class SdpaSweep(MicroBenchmark):
    NAME = "sdpa whole-kernel subblock sweep"
    ZONE = "sdpa_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/sdpa.csv"
    STRATEGIES = ("",)
    CSV_TAG = ("dtype", "fidelity", "full_sync")
    EXTRA_COLUMNS = ("qk_reuse", "out_reuse")
    POST_COLUMNS = (
        "matmul_ideal_cycles",
        "trisc_max_cycles",
        "math_cycles",
        "zone_utilization_pct",
        "math_utilization_pct",
    )
    PARAMS = (
        Param("sq", "4", help="query tiles"),
        Param("sk", "4", help="key tiles (= hd so one sub_w fits both)"),
        Param("hd", "4", help="head-dim tiles"),
        Param("sub_h", "1,2,4", sweep=True, help="subblock rows (all regions)"),
        Param("sub_w", "1,2,4", sweep=True, help="subblock cols (all regions)"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("fidelity", "hifi4", choices=("lofi", "hifi2", "hifi4")),
        Param("full_sync", False),
        Param("block_count", str(DEFAULT_BLOCK_COUNT), sweep=True),
    )
    INPUTS = (
        Tensor("q", lambda cfg: (cfg["sq"] * TILE, cfg["hd"] * TILE), scale=0.1),
        Tensor("k", lambda cfg: (cfg["sk"] * TILE, cfg["hd"] * TILE), scale=0.1),
        Tensor("v", lambda cfg: (cfg["sk"] * TILE, cfg["hd"] * TILE), scale=0.1),
    )
    OUTPUTS = (
        Tensor("o", lambda cfg: (cfg["sq"] * TILE, cfg["hd"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["sq"] * cfg["hd"]),  # q
        DFB(1, lambda cfg: cfg["sk"] * cfg["hd"]),  # k (raw)
        DFB(2, lambda cfg: cfg["sk"] * cfg["hd"]),  # v
        DFB(3, lambda cfg: cfg["sq"] * cfg["sk"]),  # s
        DFB(4, lambda cfg: 1),  # scaler
        DFB(5, lambda cfg: cfg["sq"] * cfg["sk"]),  # p
        DFB(6, lambda cfg: cfg["sq"]),  # l
        DFB(7, lambda cfg: cfg["sq"]),  # recip(l)
        DFB(8, lambda cfg: cfg["sq"]),  # recip(l) bcast
        DFB(9, lambda cfg: cfg["sq"] * cfg["hd"]),  # P@V tmp
        DFB(10, lambda cfg: cfg["hd"] * cfg["sk"]),  # kt = transpose(k)
        DFB(16, lambda cfg: cfg["block_count"] * cfg["sq"] * cfg["hd"]),  # out
    )

    def _cap(self, cfg):
        return harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )

    def legal(self, cfg, strategy):
        sh, sw = cfg["sub_h"], cfg["sub_w"]
        # one subblock drives every region: rows from Sq and (transpose) HD,
        # cols from both Sk and HD.
        if cfg["sq"] % sh or cfg["hd"] % sh or cfg["sk"] % sw or cfg["hd"] % sw:
            return False
        return sh * sw <= self._cap(cfg)

    def extra_columns(self, cfg, strategy):
        qk_reuse = (cfg["sq"] // cfg["sub_h"]) * (cfg["sk"] // cfg["sub_w"])
        out_reuse = (cfg["sq"] // cfg["sub_h"]) * (cfg["hd"] // cfg["sub_w"])
        return {"qk_reuse": qk_reuse, "out_reuse": out_reuse}

    def post_columns(self, cfg, strategy, zone_summary, row):
        tile_macs = 2 * cfg["sq"] * cfg["sk"] * cfg["hd"]
        ideal = tile_macs * MATMUL_CYCLES_PER_TILE[cfg["fidelity"]]
        freq_mhz = zone_summary["freq_mhz"]

        def cycles(us):
            return None if us is None else round(us * freq_mhz, 2)

        def util(actual):
            if actual is None or actual <= 0:
                return None
            return round(100.0 * ideal / actual, 2)

        trisc_max_cycles = cycles(zone_summary["trisc_max_us"])
        math_cycles = cycles(zone_summary["math_us"])
        return {
            "matmul_ideal_cycles": ideal,
            "trisc_max_cycles": trisc_max_cycles,
            "math_cycles": math_cycles,
            "zone_utilization_pct": util(trisc_max_cycles),
            "math_utilization_pct": util(math_cycles),
        }

    def summary(self, cfg, by_strategy):
        row = next(iter(by_strategy.values()))
        return (
            f"sq={cfg['sq']} sk={cfg['sk']} hd={cfg['hd']} "
            f"sub=({cfg['sub_h']},{cfg['sub_w']}) "
            f"reuse(qk={row['qk_reuse']},pv={row['out_reuse']}) "
            f"| trisc_max={row['trisc_max_us']} µs util={row['zone_utilization_pct']}% "
            f"pcc={row['pcc']}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        sq, sk, hd = cfg["sq"], cfg["sk"], cfg["hd"]
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
                harness.accessor_args(tensors["q"], tensors["k"], tensors["v"]),
                [
                    (
                        ctx.core,
                        [
                            tensors["q"].buffer_address(),
                            tensors["k"].buffer_address(),
                            tensors["v"].buffer_address(),
                            sq * hd,
                            sk * hd,
                            sk * hd,
                        ],
                    )
                ],
                ttnn.ReaderConfigDescriptor(),
            ),
            harness.file_kernel(
                WRITER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["o"]),
                [(ctx.core, [tensors["o"].buffer_address(), sq * hd])],
                ttnn.WriterConfigDescriptor(),
            ),
            harness.file_kernel(
                COMPUTE_KERNEL,
                ctx.grid,
                [sq, sk, hd, cfg["sub_h"], cfg["sub_w"]],
                [],
                compute,
            ),
        ]
        q = ctx.torch["q"].float()
        k = ctx.torch["k"].float()
        v = ctx.torch["v"].float()
        ref = torch.softmax(q @ k.t(), dim=-1) @ v  # K transposed in-kernel
        return kernels, ref


if __name__ == "__main__":
    SdpaSweep().main()
