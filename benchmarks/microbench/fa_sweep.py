# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Flash attention (streaming KV) with configurable matmul subblock sizes.

Single head, single core. Streams the KV sequence in chunks of `kv_chunk`
key-tiles; per chunk it does QK^T -> exp -> rowsum and P@V, accumulating the
running denominator and output across chunks (packer L1-accumulation), then
normalizes. Only one chunk's scores (Sq x kv_chunk) is ever materialized -- the
flash memory benefit -- versus `sdpa_sweep.py` which materializes the full
Sq x Sk scores. Softmax has no max-subtraction (same as the non-flash baseline;
result is identical for the bounded inputs).

Holds Sq/Sk/HD/kv_chunk fixed and sweeps the two matmuls' output subblocks
(`qk_sub_h/w`, `out_sub_h/w`). Correctness is PCC against torch
softmax(Q @ Kt) @ V. To stay in the flash regime, kv_chunk < Sk
(`kv_chunk == Sk` degenerates to single-pass / non-flash).

    python -m benchmarks.microbench.fa_sweep
    python -m benchmarks.microbench.fa_sweep --sq 4 --sk 8 --hd 4 --kv-chunk 2
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
COMPUTE_KERNEL = str(KERNELS / "fa_compute.cpp")
READER_KERNEL = str(KERNELS / "fa_reader.cpp")
WRITER_KERNEL = str(KERNELS / "drain_writer.cpp")
MATMUL_CYCLES_PER_TILE = {"lofi": 16, "hifi2": 32, "hifi4": 64}


class FaSweep(MicroBenchmark):
    NAME = "flash attention subblock sweep"
    ZONE = "fa_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/fa.csv"
    STRATEGIES = ("",)
    CSV_TAG = ("dtype", "fidelity", "full_sync")
    EXTRA_COLUMNS = ("n_chunks", "qk_reuse", "out_reuse")
    POST_COLUMNS = (
        "matmul_ideal_cycles",
        "trisc_max_cycles",
        "math_cycles",
        "zone_utilization_pct",
        "math_utilization_pct",
    )
    PARAMS = (
        Param("sq", "2", help="query tiles"),
        Param("sk", "4", help="key tiles"),
        Param("hd", "2", help="head-dim tiles"),
        Param("kv_chunk", "2", help="KV chunk tiles (flash streaming block)"),
        Param("qk_sub_h", "1,2", sweep=True, help="QK^T subblock rows"),
        Param("qk_sub_w", "1,2", sweep=True, help="QK^T subblock cols"),
        Param("out_sub_h", "1,2", sweep=True, help="P@V subblock rows"),
        Param("out_sub_w", "1,2", sweep=True, help="P@V subblock cols"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("fidelity", "hifi4", choices=("lofi", "hifi2", "hifi4")),
        Param("full_sync", False),
        Param("block_count", str(DEFAULT_BLOCK_COUNT), sweep=True),
    )
    INPUTS = (
        Tensor("q", lambda cfg: (cfg["sq"] * TILE, cfg["hd"] * TILE), scale=0.1),
        Tensor("kt", lambda cfg: (cfg["hd"] * TILE, cfg["sk"] * TILE), scale=0.1),
        Tensor("v", lambda cfg: (cfg["sk"] * TILE, cfg["hd"] * TILE), scale=0.1),
    )
    OUTPUTS = (
        Tensor("o", lambda cfg: (cfg["sq"] * TILE, cfg["hd"] * TILE), init="empty"),
    )
    DFBS = (
        DFB(0, lambda cfg: cfg["sq"] * cfg["hd"]),  # q resident
        DFB(1, lambda cfg, strategy: cfg["block_count"] * cfg["hd"] * cfg["kv_chunk"]),  # kt chunk
        DFB(2, lambda cfg, strategy: cfg["block_count"] * cfg["kv_chunk"] * cfg["hd"]),  # v chunk
        DFB(3, lambda cfg: 1),  # scaler
        DFB(4, lambda cfg: cfg["sq"] * cfg["kv_chunk"]),  # s chunk
        DFB(5, lambda cfg: cfg["sq"] * cfg["kv_chunk"]),  # p chunk
        DFB(6, lambda cfg: cfg["sq"]),  # running l
        DFB(7, lambda cfg: cfg["sq"]),  # recip(l)
        DFB(8, lambda cfg: cfg["sq"]),  # recip(l) bcast
        DFB(9, lambda cfg: cfg["sq"] * cfg["hd"]),  # running O
        DFB(16, lambda cfg: cfg["block_count"] * cfg["sq"] * cfg["hd"]),  # out
    )

    def _cap(self, cfg):
        return harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )

    def legal(self, cfg, strategy):
        if cfg["sk"] % cfg["kv_chunk"]:
            return False
        cap = self._cap(cfg)
        qh, qw = cfg["qk_sub_h"], cfg["qk_sub_w"]
        oh, ow = cfg["out_sub_h"], cfg["out_sub_w"]
        # QK^T output (sq, kv_chunk); P@V output (sq, hd).
        if cfg["sq"] % qh or cfg["kv_chunk"] % qw or qh * qw > cap:
            return False
        if cfg["sq"] % oh or cfg["hd"] % ow or oh * ow > cap:
            return False
        return True

    def extra_columns(self, cfg, strategy):
        n_chunks = cfg["sk"] // cfg["kv_chunk"]
        qk_reuse = (cfg["sq"] // cfg["qk_sub_h"]) * (
            cfg["kv_chunk"] // cfg["qk_sub_w"]
        )
        out_reuse = (cfg["sq"] // cfg["out_sub_h"]) * (cfg["hd"] // cfg["out_sub_w"])
        return {"n_chunks": n_chunks, "qk_reuse": qk_reuse, "out_reuse": out_reuse}

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
            f"sq={cfg['sq']} sk={cfg['sk']} hd={cfg['hd']} kvc={cfg['kv_chunk']}"
            f"(x{row['n_chunks']}) qk=({cfg['qk_sub_h']},{cfg['qk_sub_w']})/r{row['qk_reuse']} "
            f"pv=({cfg['out_sub_h']},{cfg['out_sub_w']})/r{row['out_reuse']} "
            f"| trisc_max={row['trisc_max_us']} µs util={row['zone_utilization_pct']}% "
            f"pcc={row['pcc']}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        sq, sk, hd, kvc = cfg["sq"], cfg["sk"], cfg["hd"], cfg["kv_chunk"]
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
                harness.accessor_args(tensors["q"], tensors["kt"], tensors["v"]),
                [
                    (
                        ctx.core,
                        [
                            tensors["q"].buffer_address(),
                            tensors["kt"].buffer_address(),
                            tensors["v"].buffer_address(),
                            sq,
                            sk,
                            hd,
                            kvc,
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
                [
                    sq,
                    sk,
                    hd,
                    kvc,
                    cfg["qk_sub_h"],
                    cfg["qk_sub_w"],
                    cfg["out_sub_h"],
                    cfg["out_sub_w"],
                ],
                [],
                compute,
            ),
        ]
        q = ctx.torch["q"].float()
        kt = ctx.torch["kt"].float()
        v = ctx.torch["v"].float()
        ref = torch.softmax(q @ kt, dim=-1) @ v
        return kernels, ref


if __name__ == "__main__":
    FaSweep().main()
