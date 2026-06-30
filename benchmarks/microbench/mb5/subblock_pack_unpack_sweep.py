# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MB1 subblock probe -- per-acquire overhead vs DST chunk size.

Same zero-compute DFB->DFB round-trip kernel as MB1 (passthrough_compute.cpp),
but instead of sweeping the block size we hold the *total* tiles fixed and sweep
the DST chunk (`sub`) moved per `tile_regs_acquire`. Each iteration still does one
DFB hop (one cb_wait_front / pop / push), so only the acquire count changes:

    acquires_per_iter = ceil(tiles / sub)

With zero arithmetic and a fixed DFB hop, this isolates the pure per-acquire
(tile_regs_acquire/commit/wait/release) overhead. It is the lightest-math regime,
so the subblock effect should be the largest and monotonic (bigger chunk -> fewer
acquires -> lower per-iter time). Contrast with the matmul/SDPA subblock sweeps,
where reuse or SFPU math reshape the picture.

    python -m benchmarks.microbench.mb5.subblock_pack_unpack_sweep --tiles 16 --sub 1,2,4,8
    python -m benchmarks.microbench.mb5.subblock_pack_unpack_sweep --tiles 16 \
        --sub 1,2,4,8,16 --full-sync 0,1
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
COMPUTE_KERNEL = str(KERNELS / "dataflow" / "passthrough_compute.cpp")
READER_KERNEL = str(KERNELS / "common" / "seed_reader.cpp")
WRITER_KERNEL = str(KERNELS / "common" / "drain_writer.cpp")


class PackUnpackChunkSweep(MicroBenchmark):
    NAME = "pack/unpack per-acquire overhead vs DST chunk"
    ZONE = "pack_unpack_loop"
    DEFAULT_CSV = "benchmarks/microbench/results/subblock_pack_unpack.csv"
    STRATEGIES = ("",)
    PER_UNIT = "iters"
    CSV_TAG = ("dtype", "full_sync", "fp32_dest_acc")
    EXTRA_COLUMNS = ("dst_chunk", "acquires")
    PARAMS = (
        Param("tiles", "16", help="total tiles per iteration (fixed)"),
        Param("sub", "1,2,4,8", sweep=True, help="DST chunk moved per acquire"),
        Param("iters", "128", help="measured iterations"),
        Param("dtype", "bf16", sweep=True, help="bf16 and/or fp32"),
        Param("full_sync", "0", sweep=True, help="dst_full_sync_en (0/1)"),
        Param("fp32_dest_acc", "0", sweep=True, help="fp32_dest_acc_en (0/1)"),
        Param(
            "block_count", str(DEFAULT_BLOCK_COUNT), sweep=True, help="DFB block count"
        ),
    )
    INPUTS = (Tensor("src", lambda cfg: (TILE, cfg["tiles"] * TILE)),)
    OUTPUTS = (Tensor("out", lambda cfg: (TILE, cfg["tiles"] * TILE), init="empty"),)
    # Only dfb_loop (the measured self-cycle) needs block_count buffering; dfb_in
    # (seed) and dfb_out (drain) are single-use outside the zone, so sizing them
    # to `tiles` frees L1 for larger totals without touching the measurement.
    DFBS = (
        DFB(0, lambda cfg: cfg["tiles"]),  # dfb_in (seed once)
        DFB(1, lambda cfg: cfg["block_count"] * cfg["tiles"]),  # dfb_loop (measured)
        DFB(16, lambda cfg: cfg["tiles"]),  # dfb_out (drain once)
    )

    def _cap(self, cfg):
        # 0/1 flags; int() so a stray non-numeric string fails loudly instead of
        # silently becoming True (bool("false") is True).
        full_sync, fp32 = bool(int(cfg["full_sync"])), bool(int(cfg["fp32_dest_acc"]))
        return harness.dst_capacity(cfg["dtype"], full_sync, fp32)

    def legal(self, cfg, strategy):
        sub = cfg["sub"]
        # an acquire holds at most the DST capacity, and a chunk past `tiles`
        # just duplicates sub == tiles, so cap the useful sweep there.
        return 0 < sub <= self._cap(cfg) and sub <= cfg["tiles"]

    def extra_columns(self, cfg, strategy):
        sub = cfg["sub"]
        return {"dst_chunk": sub, "acquires": -(-cfg["tiles"] // sub)}

    def summary(self, cfg, by_strategy):
        row = next(iter(by_strategy.values()))
        per_iter = row["trisc_max_us_per_iters"]
        per_iter = "n/a" if per_iter is None else f"{per_iter:.4f}"
        return (
            f"tiles={cfg['tiles']} chunk={row['dst_chunk']} "
            f"acquires={row['acquires']} dtype={cfg['dtype']} "
            f"fullsync={cfg['full_sync']} fp32={cfg['fp32_dest_acc']} "
            f"| trisc_max={row['trisc_max_us']} µs per_iter={per_iter} µs "
            f"pcc={row['pcc']}"
        )

    def build(self, ctx):
        cfg = ctx.cfg
        tiles, iters = cfg["tiles"], cfg["iters"]
        full_sync, fp32 = bool(int(cfg["full_sync"])), bool(int(cfg["fp32_dest_acc"]))
        cap = cfg["sub"]  # the swept subblock IS the per-acquire DST chunk
        tensors = ctx.tensors
        compute = harness.compute_config(fp32_dest_acc=fp32, full_sync=full_sync)
        kernels = [
            harness.file_kernel(
                READER_KERNEL,
                ctx.grid,
                harness.accessor_args(tensors["src"]),
                [(ctx.core, [tensors["src"].buffer_address(), tiles])],
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
                COMPUTE_KERNEL, ctx.grid, [tiles, iters, cap], [], compute
            ),
        ]
        return kernels, ctx.torch["src"]


if __name__ == "__main__":
    PackUnpackChunkSweep().main()
