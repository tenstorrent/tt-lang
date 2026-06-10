# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""out = a*x + b*y over exactly ONE block, compute-isolated.

The fused multiply-add from ``test/python/test_axby.py``: four separate inputs,
two independent multiply intermediates live in DST at once, then summed -- the
register-allocation stressor (the original bug computed (a*x)*b instead of b*y).

Stripped to the bare compute (same style as single_block_add): the compute thread
*reserves* all five blocks itself (inputs are uninitialized L1 -- correctness is
irrelevant, only the compute cycles are) and the data-movement threads do nothing,
so the measured cycles are pure compute.
"""

from __future__ import annotations

import ttl


def make_single_block_axby_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block axby: out = a*x + b*y."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __axby_single_block_no_dram(a, x, b, y, out) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=2)
        x_dfb = ttl.make_dataflow_buffer_like(x, shape=(R, C), block_count=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=2)
        y_dfb = ttl.make_dataflow_buffer_like(y, shape=(R, C), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(R, C), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as av,
                x_dfb.reserve() as xv,
                b_dfb.reserve() as bv,
                y_dfb.reserve() as yv,
                out_dfb.reserve() as o,
            ):
                term1 = av * xv
                term2 = bv * yv
                o.store(term1 + term2)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __axby_single_block_no_dram
