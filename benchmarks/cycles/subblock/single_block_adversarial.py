# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Adversarial fused elementwise over exactly ONE block, compute-isolated.

A single core does one (row_tiles_per_block x col_tiles_per_block)-tile block of
the 4-input/4-output adversarial kernel from
``test/python/test_adversarial_multinode.py`` -- four interleaved tanh/sigmoid/
abs/relu/neg chains sharing ``sigmoid(c)``:

  out1 = abs(sigmoid( tanh(a) + b + sigmoid(c) ))
  out2 = abs(neg(relu( sigmoid(c) + tanh(b) )))
  out3 = relu(sigmoid(tanh( sigmoid(c) + d )))
  out4 = sigmoid(abs( tanh(c) + d ))

Stripped to the bare compute (same style as single_block_add): the compute thread
*reserves* all eight blocks itself (inputs are uninitialized L1 -- correctness is
irrelevant, only the compute cycles are) and the data-movement threads do nothing,
so the measured cycles are pure compute. The original nesting is kept (out1/out2
share one output scope, out3/out4 another) since it shapes the DST pressure.
"""

from __future__ import annotations

import ttnn
import ttl


def make_single_block_adversarial_no_dram(
    *,
    row_tiles_per_block: int,
    col_tiles_per_block: int,
    grid=(1, 1),
    fp32_dest_acc_en: bool = False,
    dst_full_sync_en: bool = False,
    compiler_options: str | None = None,
):
    """Compute-isolated single-block adversarial fused kernel (4 in, 4 out)."""
    decorator_kwargs = dict(
        grid=grid, fp32_dest_acc_en=fp32_dest_acc_en, dst_full_sync_en=dst_full_sync_en
    )
    if compiler_options is not None:
        decorator_kwargs["options"] = compiler_options

    R, C = row_tiles_per_block, col_tiles_per_block

    @ttl.operation(**decorator_kwargs)
    def __adversarial_single_block_no_dram(a, b, c, d, out1, out2, out3, out4) -> None:
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(R, C), block_count=1)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(R, C), block_count=1)
        c_dfb = ttl.make_dataflow_buffer_like(c, shape=(R, C), block_count=1)
        d_dfb = ttl.make_dataflow_buffer_like(d, shape=(R, C), block_count=1)
        out1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(R, C), block_count=1)
        out2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(R, C), block_count=1)
        out3_dfb = ttl.make_dataflow_buffer_like(out3, shape=(R, C), block_count=1)
        out4_dfb = ttl.make_dataflow_buffer_like(out4, shape=(R, C), block_count=1)

        # Compute owns all blocks (no producer/consumer handshake): reserve the
        # inputs (uninitialized L1, don't-care) and outputs, run the chains, store.
        @ttl.compute()
        def compute():
            with (
                a_dfb.reserve() as av,
                b_dfb.reserve() as bv,
                c_dfb.reserve() as cv,
                d_dfb.reserve() as dv,
            ):
                with (out1_dfb.reserve() as o1, out2_dfb.reserve() as o2):
                    # out1: sigmoid(c) + a + b chain
                    v = ttl.math.tanh(av)
                    v = v + bv
                    v = v + ttl.math.sigmoid(cv)
                    v = ttl.math.sigmoid(v)
                    v = ttl.math.abs(v)
                    o1.store(v)

                    # out2: operands in reverse order
                    v = ttl.math.tanh(bv)
                    v = ttl.math.sigmoid(cv) + v
                    v = ttl.math.relu(v)
                    v = ttl.math.neg(v)
                    v = ttl.math.abs(v)
                    o2.store(v)

                with (out3_dfb.reserve() as o3, out4_dfb.reserve() as o4):
                    # out3: chain using sigmoid(c) + d
                    v = ttl.math.sigmoid(cv) + dv
                    v = ttl.math.tanh(v)
                    v = ttl.math.sigmoid(v)
                    v = ttl.math.relu(v)
                    o3.store(v)

                    # out4: different combination
                    v = ttl.math.tanh(cv)
                    v = v + dv
                    v = ttl.math.abs(v)
                    v = ttl.math.sigmoid(v)
                    o4.store(v)

        @ttl.datamovement()
        def read():
            pass  # no DRAM read, no CB handshake

        @ttl.datamovement()
        def write():
            pass  # no DRAM write, no CB handshake

    return __adversarial_single_block_no_dram
