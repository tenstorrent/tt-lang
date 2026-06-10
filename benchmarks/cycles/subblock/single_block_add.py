# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""y = a + b over exactly ONE block, with no grid/node for-loops.

A single core does a single (row_tiles_per_block x col_tiles_per_block)-tile
block: the body is one wait/add/store, no `for local_row/local_col` loops and no
node distribution. `y = a + b` is dstPerIteration = 1, so when the block fits the
DST budget it runs in a single acquire/add/pack sync region.

  - make_single_block_add          : real DRAM read/add/write (correctness).
  - make_single_block_add_no_dram  : compute-isolated (data movement gutted);
                                     used by the cycle benchmark.
"""

from __future__ import annotations

import ttnn
import ttl


def make_single_block_add(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Real single-block add: read one block, add, write one block. No loops."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __add_single_block(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                y_dfb.reserve() as y_blk,
            ):
                y_blk.store(a_blk + b_blk)

        @ttl.datamovement()
        def read():
            with (a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk):
                tx_a = ttl.copy(a[0:R, 0:C], a_blk)
                tx_b = ttl.copy(b[0:R, 0:C], b_blk)
                tx_a.wait()
                tx_b.wait()

        @ttl.datamovement()
        def write():
            with y_dfb.wait() as y_blk:
                tx = ttl.copy(y_blk, y[0:R, 0:C])
                tx.wait()

    return __add_single_block


def make_single_block_add_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block add, stripped to the bare compute. The
    compute thread *reserves* all three blocks itself and does the add (a/b hold
    uninitialized L1 -- correctness is irrelevant, only the compute cycles are),
    so there is no CB handshake at all: the data-movement threads do nothing (no
    reserve/wait), leaving the measured cycles as pure compute."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __add_single_block_no_dram(a: ttnn.Tensor, b: ttnn.Tensor, y: ttnn.Tensor) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(R, C), block_count=2)

        # Compute owns all three blocks (no producer/consumer handshake): reserve
        # a/b/y, add, store. a_blk/b_blk are uninitialized L1 -- we only measure
        # the compute, so the data is don't-care.
        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as a_blk,
                b_dfb.reserve() as b_blk,
                y_dfb.reserve() as y_blk,
            ):
                y_blk.store(a_blk + b_blk)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __add_single_block_no_dram