# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""fill kernels over exactly ONE block, compute-isolated.

The two fill patterns from ``test/python/test_fill.py``:

  fill_add : out = inp + fill(1.0)   fill into DST, copy input, SFPU add.
  fill     : out = fill(-3.0)        pure fill + pack -- the shallowest
                                     possible compute kind (no inputs at all;
                                     the original's dm_read is already a no-op).

Stripped to the bare compute (same style as single_block_add): the compute
thread *reserves* every block itself (inputs are uninitialized L1 -- correctness
is irrelevant, only the compute cycles are) and the data-movement threads do
nothing, so the measured cycles are pure compute.
"""

from __future__ import annotations

import ttl


def make_single_block_fill_add_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block fill+add: out = inp + fill(1.0)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __fill_add_single_block_no_dram(inp, out) -> None:
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (inp_dfb.reserve() as x, out_dfb.reserve() as o):
                o.store(x + ttl.block.fill(1.0, shape=o.shape))

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __fill_add_single_block_no_dram


def make_single_block_fill_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block pure fill: out = fill(-3.0)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __fill_single_block_no_dram(out) -> None:
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with out_dfb.reserve() as o:
                o.store(ttl.block.fill(-3.0, shape=o.shape))

        @ttl.datamovement()
        def read():
            pass  # no inputs at all

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __fill_single_block_no_dram
