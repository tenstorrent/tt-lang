# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Broadcast (one-to-many multicast) pipe under receiver-DFB pipelining.

One sender multicasts a tile to N receivers, each of which reserves, receives,
and writes the tile to its own output column. Because a single ``net.if_dst``
callback spans the whole receiver range, the pop's static domain is the union of
receiver nodes, so lowering keeps receiver-ready synchronization rather than
sender-local capacity. recv_block_count > 1 exercises dynamic computed multicast
addresses with the receiver running several blocks deep.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32
N_RECV = 3
N_ITERS = 8


def _broadcast_op(recv_block_count):
    @ttl.operation(grid=(N_RECV + 1, 1))
    def broadcast_loop(inp, out):
        net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, N_RECV + 1), 0))])
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(1, 1), block_count=recv_block_count
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm():
            node_x, _node_y = ttl.node(dims=2)
            for _iter_idx in range(N_ITERS):

                def send(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                net.if_src(send)

                def recv(pipe):
                    with recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()
                    with recv_dfb.wait() as recv_blk:
                        ttl.copy(recv_blk, out[0, node_x]).wait()

                net.if_dst(recv)

        @ttl.datamovement()
        def dm_brisc():
            pass

    return broadcast_loop


# Repeated multicast must preserve data while the receiver advances through a
# multi-block DFB.
@pytest.mark.parametrize("recv_block_count", [2, 3], ids=["bc2", "bc3"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_broadcast_loop(device, dtype, recv_block_count):
    inp_torch = torch.randn(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, (N_RECV + 1) * TILE, dtype=dtype)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    _broadcast_op(recv_block_count)(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out).float()
    for col in range(1, N_RECV + 1):
        assert_pcc(result[:, col * TILE : (col + 1) * TILE], inp_torch.float())
