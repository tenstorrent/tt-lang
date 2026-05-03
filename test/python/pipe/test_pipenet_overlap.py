# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Overlapping multicast destinations in one PipeNet (issue #505).

Two senders at (0, 0) and (1, 0) multicast to receivers at (2, 0) and
(3, 0). Each receiver gets two tiles in two CB blocks and sums them.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32


@ttl.operation(grid=(4, 1))
def overlap_sum_kernel(inp, out):
    """Senders (0, 0) and (1, 0) each multicast one tile to receivers
    (2, 0) and (3, 0). Each receiver sums the two received tiles.
    """
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(slice(2, 4), 0)),
            ttl.Pipe(src=(1, 0), dst=(slice(2, 4), 0)),
        ]
    )

    # Receiver CB needs block_count >= 2 (one slot per sender).
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    acc_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        # Only receivers run compute.
        if x >= 2:
            # First sender's tile lands in slot 0; second in slot 1.
            with recv_cb.wait() as t, acc_cb.reserve() as a:
                a.store(t)
            with recv_cb.wait() as t, acc_cb.wait() as prev, out_cb.reserve() as o:
                o.store(prev + t)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        if x < 2:
            # Sender: stage own input tile and send.
            with recv_cb.reserve() as blk:
                ttl.copy(inp[0, x], blk).wait()

                def send(pipe):
                    ttl.copy(blk, pipe).wait()

                net.if_src(send)
        else:
            # Receiver: each pipe targeting this core reserves its own CB
            # block. With block_count >= number of senders, the two senders
            # land in distinct blocks.
            def recv(pipe):
                with recv_cb.reserve() as blk:
                    ttl.copy(pipe, blk).wait()

            net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        if x >= 2:
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, x - 2]).wait()


def test_overlapping_multicast(device):
    """Two senders multicasting to two shared receivers; each receiver
    sums the two tiles."""
    inp_torch = torch.randn(TILE, 2 * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16), device)

    overlap_sum_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    tile_sum = (
        inp_torch[:, 0:TILE].float() + inp_torch[:, TILE : 2 * TILE].float()
    ).to(torch.bfloat16)
    expected = tile_sum.repeat(1, 2)
    assert_pcc(expected, result)
