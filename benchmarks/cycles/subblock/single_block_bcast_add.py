# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Column-broadcast add over exactly ONE block, compute-isolated.

A single core does one (row_tiles_per_block x col_tiles_per_block)-tile block of
the broadcast-add from ``test/python/test_bcast_add.py``:

  out = ttl.block.broadcast(b, dims=[1], shape=(R, C)) + a

``b`` is a tile-COLUMN (shape (R, 1)) replicated across the C output tile-columns
(``tile_bcast`` col), then added to the full (R, C) block ``a``. The broadcast
writes a DST tile and the add (``add_binary_tile``) combines it with ``a``, so the
forced subblock (sR, sC) is bounded by the DST budget the compiler enforces.

Stripped to the bare compute (same style as single_block_add): the compute thread
*reserves* all three blocks itself (inputs are uninitialized L1 -- correctness is
irrelevant, only the compute cycles are) and the data-movement threads do nothing,
so the measured cycles are pure compute.
"""

from __future__ import annotations

import ttl


def make_single_block_bcast_add_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block column-broadcast add: out = bcast_col(b) + a."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __bcast_add_single_block_no_dram(a, b, out) -> None:
        # b is a tile-column (R, 1); a and out are full (R, C) blocks.
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        # Compute owns all three blocks (no producer/consumer handshake): reserve
        # a/b/out (a/b uninitialized L1, don't-care), broadcast b across columns,
        # add a, store.
        @ttl.compute()
        def compute():
            with (
                b_dfb.reserve() as b_tile,
                a_dfb.reserve() as a_tile,
                out_dfb.reserve() as o,
            ):
                b_bcast = ttl.block.broadcast(b_tile, dims=[1], shape=(R, C))
                o.store(b_bcast + a_tile)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __bcast_add_single_block_no_dram
