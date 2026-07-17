# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""RA/RP, CA/RP, and CA/CC produce the same result.

The point-to-point pipe is eligible for computed receiver addresses by default;
--no-ttl-pipe-capacity-sync selects receiver-post synchronization, and
--no-ttl-pipe-computed-addresses selects receiver-published addresses. All
three protocols must match torch and each other across dtypes. A receiver block
count greater than one exercises dynamic sender-side slot indices.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32
N_ITERS = 4


def _point_to_point(inp, out, recv_block_count):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
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
                    ttl.copy(recv_blk, out[0, 0]).wait()

            if node_x == 1:
                net.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


def _make_point_to_point_ops(recv_block_count):
    @ttl.operation(grid=(2, 1))
    def computed_capacity_counter(inp, out):
        _point_to_point(inp, out, recv_block_count)

    @ttl.operation(grid=(2, 1), options="--no-ttl-pipe-capacity-sync")
    def computed_receiver_post(inp, out):
        _point_to_point(inp, out, recv_block_count)

    @ttl.operation(grid=(2, 1), options="--no-ttl-pipe-computed-addresses")
    def receiver_address_receiver_post(inp, out):
        _point_to_point(inp, out, recv_block_count)

    return (
        computed_capacity_counter,
        computed_receiver_post,
        receiver_address_receiver_post,
    )


@pytest.mark.parametrize("recv_block_count", [1, 2], ids=["bc1", "bc2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_pipe_protocols_match(device, dtype, recv_block_count):
    pipe_operations = _make_point_to_point_ops(recv_block_count)
    inp_torch = torch.randn(TILE, TILE, dtype=dtype)

    protocol_results = []
    for pipe_operation in pipe_operations:
        output = to_dram(torch.zeros(TILE, TILE, dtype=dtype), device)
        pipe_operation(to_dram(inp_torch, device), output)
        ttnn.synchronize_device(device)
        protocol_results.append(ttnn.to_torch(output).float())

    for protocol_result in protocol_results:
        assert_pcc(protocol_result, inp_torch.float())
    for protocol_result in protocol_results[1:]:
        assert_pcc(protocol_results[0], protocol_result)
