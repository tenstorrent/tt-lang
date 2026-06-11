# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reduce over exactly ONE block, compute-isolated.

The single-core reduce from ``test/python/test_reduce.py``: out = reduce_fn(inp,
dims=[1]) on one (R, C)-tile block, producing an (R, 1)-tile column. The
reduction dim keeps its full size inside each DST region, so the forced subblock
is one-dimensional: --ttl-force-subblock=sR (single entry to match the single
parallel dim).

The tracer cannot capture function- or bool-typed closure vars, so sum and max
are separate literal kernels (no dynamic dispatch in the compute body).

Stripped to the bare compute (same style as single_block_add): the compute
thread *reserves* both blocks itself (input is uninitialized L1 -- correctness
is irrelevant, only the compute cycles are) and the data-movement threads do
nothing, so the measured cycles are pure compute.
"""

from __future__ import annotations

import ttl


def _kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options):
    kw = dict(grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en)
    if compiler_options is not None:
        kw["options"] = compiler_options
    return kw


def make_single_block_reduce_sum_no_dram(
    *, row_tiles_per_block, col_tiles_per_block, grid=(1, 1),
    fp32_dest_acc_en=False, dst_full_sync_en=False, compiler_options=None,
):
    """Compute-isolated single-block row reduce: out = reduce_sum(inp, dims=[1])."""
    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options))
    def __reduce_sum_single_block_no_dram(inp, out) -> None:
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, 1), block_count=2)

        @ttl.compute()
        def compute():
            with (inp_dfb.reserve() as x, out_dfb.reserve() as o):
                o.store(ttl.math.reduce_sum(x, dims=[1]))

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __reduce_sum_single_block_no_dram


def make_single_block_reduce_max_no_dram(
    *, row_tiles_per_block, col_tiles_per_block, grid=(1, 1),
    fp32_dest_acc_en=False, dst_full_sync_en=False, compiler_options=None,
):
    """Compute-isolated single-block row reduce: out = reduce_max(inp, dims=[1])."""
    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**_kwargs(grid, fp32_dest_acc_en, dst_full_sync_en, compiler_options))
    def __reduce_max_single_block_no_dram(inp, out) -> None:
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, 1), block_count=2)

        @ttl.compute()
        def compute():
            with (inp_dfb.reserve() as x, out_dfb.reserve() as o):
                o.store(ttl.math.reduce_max(x, dims=[1]))

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __reduce_max_single_block_no_dram
