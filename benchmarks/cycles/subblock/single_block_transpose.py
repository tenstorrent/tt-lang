# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Transpose over exactly ONE block, compute-isolated.

The single-core transpose from ``test/python/test_transpose.py``:
out = transpose(inp) on one (R, C)-tile block producing a (C, R)-tile block
(both block dims parallel -- the transpose is per-tile + tile permutation, no
reduction). Defaults to a square 8x8 block so input and output share geometry.

Stripped to the bare compute (same style as single_block_add): the compute
thread *reserves* both blocks itself (input is uninitialized L1 -- correctness
is irrelevant, only the compute cycles are) and the data-movement threads do
nothing, so the measured cycles are pure compute.
"""

from __future__ import annotations

import ttl


def make_single_block_transpose_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block transpose: out = transpose(inp)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __transpose_single_block_no_dram(inp, out) -> None:
        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(C, R), block_count=2)

        @ttl.compute()
        def compute():
            with (inp_dfb.reserve() as x, out_dfb.reserve() as o):
                o.store(ttl.math.transpose(x))

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __transpose_single_block_no_dram
