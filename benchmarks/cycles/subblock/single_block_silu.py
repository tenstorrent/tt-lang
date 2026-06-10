# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SiLU on a DST intermediate over exactly ONE block, compute-isolated.

A single core does one (row_tiles_per_block x col_tiles_per_block)-tile block of
the SiLU kernel from ``test/python/test_copy_dst.py`` (issue #443):

  y = a + b;  out = y * sigmoid(y)

``y`` is a DST intermediate referenced twice; ``sigmoid`` is a destructive
in-place SFPU unary, so the compiler inserts ``ttl.copy_dst`` to preserve ``y``
for the outer multiply. Each output tile therefore holds y, its copy, and the
product in DST -- a heavier dstPerIteration than the plain SFPU chains, probing
the copy_dst path under subblocking.

Stripped to the bare compute (same style as single_block_add): the compute thread
*reserves* all three blocks itself (inputs are uninitialized L1 -- correctness is
irrelevant, only the compute cycles are) and the data-movement threads do nothing,
so the measured cycles are pure compute.
"""

from __future__ import annotations

import ttl


def make_single_block_silu_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block SiLU on a DST intermediate (copy_dst path)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __silu_single_block_no_dram(a, b, out) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        # Compute owns all three blocks (no producer/consumer handshake): reserve
        # a/b/out (a/b uninitialized L1, don't-care), SiLU, store.
        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as av,
                b_dfb.reserve() as bv,
                out_dfb.reserve() as o,
            ):
                y = av + bv
                o.store(y * ttl.math.sigmoid(y))

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __silu_single_block_no_dram
