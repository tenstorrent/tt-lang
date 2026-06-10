# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""x * rsqrt(abs(x)) on a DST intermediate over ONE block, compute-isolated.

The other copy_dst kernel from ``test/python/test_copy_dst.py`` (issue #384):

  x = a * b;  out = x * rsqrt(abs(x))

``x`` is a DST intermediate referenced twice; abs/rsqrt are destructive in-place
SFPU unaries chained on the copy, so the compiler emits ttl.copy_dst to preserve
``x`` for the outer multiply. Two chained unaries on the copy vs silu's one.

Stripped to the bare compute (same style as single_block_add): the compute thread
*reserves* all three blocks itself (inputs are uninitialized L1 -- correctness is
irrelevant, only the compute cycles are) and the data-movement threads do nothing,
so the measured cycles are pure compute.
"""

from __future__ import annotations

import ttl


def make_single_block_rsqrt_abs_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block x * rsqrt(abs(x)) (copy_dst path)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __rsqrt_abs_single_block_no_dram(a, b, out) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as av,
                b_dfb.reserve() as bv,
                out_dfb.reserve() as o,
            ):
                x = av * bv
                o.store(x * ttl.math.rsqrt(ttl.math.abs(x)))

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __rsqrt_abs_single_block_no_dram
