# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Subblock-selection probe: how the subblock shape affects a matmul, for one
K-accumulation strategy (selected with --accum, default dst).

Holds the output fixed (default 8x8 tiles, the "clean" power-of-two case) and
sweeps every valid (sub_mt, sub_nt) subblock -- each shape dividing (mt, nt)
with sub_mt*sub_nt <= DST capacity. The kernel is picked by --accum:

  dst (default): matmul_k_dst.cpp -- the subblock stays resident in DST across
       the K loop, packed once; operands are re-unpacked per subblock (reuse).
       Acquire count = reuse, so the subblock washes out as kt grows.
  l1:  matmul_k_l1.cpp -- each K step matmuls every subblock into a fresh DST and
       packs to L1 with packer accumulation. Acquire count = reuse*kt, so the
       subblock stays a large lever at every kt.

Both use the same resident operands (L1-K just pops mt/nt per K step), so the only
variable per config is the subblock. The `compiler_pick` column flags the subblock
the compiler would choose (harness.dst_subblock == computeMultiDimSubblockSizes);
compare it against the fastest row.

    # 8x8 output, all valid subblocks, K-depths 1/2/4/8 (DST-K, default):
    python -m benchmarks.microbench.mb5.subblock_matmul_sweep

    # L1 accumulation instead:
    python -m benchmarks.microbench.mb5.subblock_matmul_sweep --accum l1

    # focus a single depth:
    python -m benchmarks.microbench.mb5.subblock_matmul_sweep --kt 8
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
DST_KERNEL = str(KERNELS / "matmul" / "matmul_k_dst.cpp")
L1_KERNEL = str(KERNELS / "matmul" / "matmul_k_l1.cpp")
READER_KERNEL = str(KERNELS / "matmul" / "matmul_reader.cpp")
WRITER_KERNEL = str(KERNELS / "common" / "drain_writer.cpp")
MATMUL_CYCLES_PER_TILE = {"lofi": 16, "hifi2": 32, "hifi4": 64}


class SubblockSweep(MicroBenchmark):
    NAME = "matmul subblock selection"
    ZONE = "matmul_k_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/subblock_matmul.csv"
    STRATEGIES = ("",)  # one accumulation strategy per run, chosen by --accum
    PER_UNIT = "kt"
    CSV_TAG = ("dtype", "fidelity", "full_sync", "accum")
    EXTRA_COLUMNS = ("reuse", "compiler_pick")
    POST_COLUMNS = (
        "matmul_ideal_cycles",
        "trisc_max_cycles",
        "math_cycles",
        "zone_utilization_pct",
        "math_utilization_pct",
    )
    PARAMS = (
        Param("mt", "8", help="output rows (tiles)"),
        Param("nt", "8", help="output cols (tiles)"),
        Param("kt", "1,2,4,8", sweep=True, help="K-depth (tiles)"),
        Param("sub_mt", "1,2,4,8", sweep=True, help="subblock rows (tiles)"),
        Param("sub_nt", "1,2,4,8", sweep=True, help="subblock cols (tiles)"),
        Param("accum", "dst", choices=("dst", "l1"), help="K-accumulation strategy"),
        Param("dtype", "bf16", choices=("bf16", "fp32")),
        Param("fidelity", "hifi4", choices=("lofi", "hifi2", "hifi4")),
        Param("full_sync", False),
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
    # DST-K keeps both operands resident (matmul_k_dst.cpp: dfb 0/1/16).
    DFBS = (
        DFB(0, lambda cfg: cfg["kt"] * cfg["mt"]),
        DFB(1, lambda cfg: cfg["kt"] * cfg["nt"]),
        DFB(16, lambda cfg: cfg["block_count"] * cfg["mt"] * cfg["nt"]),
    )

    def _cap(self, cfg):
        return harness.dst_capacity(
            cfg["dtype"], cfg["full_sync"], cfg["dtype"] == "fp32"
        )

    def legal(self, cfg, strategy):
        """Only valid subblocks: divide (mt, nt) and fit DST capacity."""
        sub_mt, sub_nt = cfg["sub_mt"], cfg["sub_nt"]
        if cfg["mt"] % sub_mt or cfg["nt"] % sub_nt:
            return False
        return sub_mt * sub_nt <= self._cap(cfg)

    def extra_columns(self, cfg, strategy):
        sub_mt, sub_nt = cfg["sub_mt"], cfg["sub_nt"]
        reuse = (cfg["mt"] // sub_mt) * (cfg["nt"] // sub_nt)
        pick = harness.dst_subblock(cfg["mt"], cfg["nt"], self._cap(cfg))
        return {"reuse": reuse, "compiler_pick": int((sub_mt, sub_nt) == pick)}

    def post_columns(self, cfg, strategy, zone_summary, row):
        ideal = cfg["mt"] * cfg["nt"] * cfg["kt"] * MATMUL_CYCLES_PER_TILE[cfg["fidelity"]]
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
        star = "  <== compiler pick" if row["compiler_pick"] else ""
        return (
            f"[{cfg['accum']}] mt={cfg['mt']} nt={cfg['nt']} kt={cfg['kt']} "
            f"sub=({cfg['sub_mt']},{cfg['sub_nt']}) reuse={row['reuse']:>2} "
            f"| trisc_max={row['trisc_max_us']} µs "
            f"unpack={row['unpack_us']} math={row['math_us']} pack={row['pack_us']} "
            f"| util={row['zone_utilization_pct']}% pcc={row['pcc']}{star}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        mt, nt, kt = cfg["mt"], cfg["nt"], cfg["kt"]
        sub_mt, sub_nt = cfg["sub_mt"], cfg["sub_nt"]
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
                DST_KERNEL if cfg["accum"] == "dst" else L1_KERNEL,
                ctx.grid,
                [mt, nt, kt, sub_mt, sub_nt, 0],  # fuse = 0 (plain matmul)
                [],
                compute,
            ),
        ]
        ref = ctx.torch["a"].float() @ ctx.torch["b"].float()
        return kernels, ref


if __name__ == "__main__":
    SubblockSweep().main()
