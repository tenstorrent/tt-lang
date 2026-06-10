# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive fused elementwise over exactly ONE block, compute-isolated.

A single core does one (row_tiles_per_block x col_tiles_per_block)-tile block of
the 3-input/3-output kernel from ``test/python/test_comprehensive_multinode.py``
-- 20 fused sigmoid/tanh/abs/relu/neg ops across three chains:

  out1 = relu(abs(tanh(sigmoid( tanh(sigmoid(a)) + b ))))
  out2 = sigmoid(abs(neg(tanh( sigmoid(tanh(b)) + c ))))
  out3 = sigmoid(abs(tanh( sigmoid(relu(a)) + c )))

Unlike adversarial, ALL THREE outputs share one flat scope (the original kernel
reserves o1/o2/o3 together), which shapes the DST pressure differently.

Stripped to the bare compute (same style as single_block_add): the compute thread
*reserves* all six blocks itself (inputs are uninitialized L1 -- correctness is
irrelevant, only the compute cycles are) and the data-movement threads do nothing,
so the measured cycles are pure compute. block_count=1: six double-buffered 8x8
CBs would exceed the ~1.43 MB L1 budget (6 * 64 * 2 * 2048 B = 1.5 MB); single
buffering halves that and there is nothing to pipeline anyway.
"""

from __future__ import annotations

import ttl


def make_single_block_comprehensive_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block comprehensive fused kernel (3 in, 3 out)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __comprehensive_single_block_no_dram(a, b, c, out1, out2, out3) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=1)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=1)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(R, C), block_count=1)
        out1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(R, C), block_count=1)
        out2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(R, C), block_count=1)
        out3_dfb = ttl.make_dataflow_buffer_like(out3, shape=(R, C), block_count=1)

        # Compute owns all six blocks (no producer/consumer handshake): reserve
        # the inputs (uninitialized L1, don't-care) and outputs in ONE flat scope
        # (as in the source test), run the three chains, store.
        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as av,
                b_dfb.reserve() as bv,
                c_dfb.reserve() as cv,
                out1_dfb.reserve() as o1,
                out2_dfb.reserve() as o2,
                out3_dfb.reserve() as o3,
            ):
                # out1 = f(a, b): 7 ops
                v1 = ttl.math.sigmoid(av)
                v1 = ttl.math.tanh(v1)
                v1 = v1 + bv
                v1 = ttl.math.sigmoid(v1)
                v1 = ttl.math.tanh(v1)
                v1 = ttl.math.abs(v1)
                v1 = ttl.math.relu(v1)
                o1.store(v1)

                # out2 = g(b, c): 7 ops
                v2 = ttl.math.tanh(bv)
                v2 = ttl.math.sigmoid(v2)
                v2 = v2 + cv
                v2 = ttl.math.tanh(v2)
                v2 = ttl.math.neg(v2)
                v2 = ttl.math.abs(v2)
                v2 = ttl.math.sigmoid(v2)
                o2.store(v2)

                # out3 = h(a, c): 6 ops
                v3 = ttl.math.relu(av)
                v3 = ttl.math.sigmoid(v3)
                v3 = v3 + cv
                v3 = ttl.math.tanh(v3)
                v3 = ttl.math.abs(v3)
                v3 = ttl.math.sigmoid(v3)
                o3.store(v3)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __comprehensive_single_block_no_dram
