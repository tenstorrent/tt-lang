# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipe rendezvous coverage for posted receives.

These tests cover cases where the receiver publishes one or more destination
DFB addresses before waiting for the transfers to complete.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

TILE = 32


def make_two_net_posted_gather_kernel():
    first_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    second_pipe = ttl.Pipe(src=(1, 0), dst=(2, 0))
    first_net = ttl.PipeNet([first_pipe])
    second_net = ttl.PipeNet([second_pipe])

    @ttl.operation(grid=(3, 1))
    def posted_gather(inp, out):
        _first_net = first_net
        _second_net = second_net

        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 2:
                with recv_dfb.wait() as first_recv_blk, acc_dfb.reserve() as acc_blk:
                    acc_blk.store(first_recv_blk)
                with (
                    recv_dfb.wait() as second_recv_blk,
                    acc_dfb.wait() as acc_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    out_blk.store(acc_blk + second_recv_blk)

        @ttl.datamovement()
        def post_receives_and_send():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                    ttl.copy(send_blk, first_pipe).wait()
            if node_x == 1:
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 1], send_blk).wait()
                    ttl.copy(send_blk, second_pipe).wait()
            if node_x == 2:
                with (
                    recv_dfb.reserve() as first_dst_blk,
                    recv_dfb.reserve() as second_dst_blk,
                ):
                    first_recv_tx = ttl.copy(first_pipe, first_dst_blk)
                    second_recv_tx = ttl.copy(second_pipe, second_dst_blk)
                    first_recv_tx.wait()
                    second_recv_tx.wait()

        @ttl.datamovement()
        def write_output():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 2:
                with out_dfb.wait() as out_blk:
                    ttl.copy(out_blk, out[0, 0]).wait()

    return posted_gather


def make_same_source_two_pipe_kernel():
    first_pipe = ttl.Pipe(src=(0, 0), dst=(1, 0))
    second_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    same_source_net = ttl.PipeNet([first_pipe, second_pipe])

    @ttl.operation(grid=(3, 1))
    def same_source_two_pipe(inp, out):
        _same_source_net = same_source_net

        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 1 or node_x == 2:
                with recv_dfb.wait() as recv_blk, out_dfb.reserve() as out_blk:
                    out_blk.store(recv_blk)

        @ttl.datamovement()
        def post_receive_and_send():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with send_dfb.reserve() as first_send_blk:
                    ttl.copy(inp[0, 0], first_send_blk).wait()
                    ttl.copy(first_send_blk, first_pipe).wait()
                with send_dfb.reserve() as second_send_blk:
                    ttl.copy(inp[0, 1], second_send_blk).wait()
                    ttl.copy(second_send_blk, second_pipe).wait()
            if node_x == 1:
                with recv_dfb.reserve() as first_dst_blk:
                    ttl.copy(first_pipe, first_dst_blk).wait()
            if node_x == 2:
                with recv_dfb.reserve() as second_dst_blk:
                    ttl.copy(second_pipe, second_dst_blk).wait()

        @ttl.datamovement()
        def write_output():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 1 or node_x == 2:
                with out_dfb.wait() as out_blk:
                    ttl.copy(out_blk, out[0, node_x - 1]).wait()

    return same_source_two_pipe


posted_gather_kernel = make_two_net_posted_gather_kernel()
same_source_two_pipe_kernel = make_same_source_two_pipe_kernel()


def make_non_uniform_multicast_receive_address_kernel():
    bcast_pipe = ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))
    bcast_net = ttl.PipeNet([bcast_pipe])

    @ttl.operation(grid=(3, 1))
    def non_uniform_multicast_receive_address(inp, out):
        _bcast_net = bcast_net

        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def post_receive_and_send():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 0:
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                    ttl.copy(send_blk, bcast_pipe).wait()
            if node_x == 1 or node_x == 2:
                with recv_dfb.reserve() as recv_blk:

                    def recv(pipe):
                        if node_x == 1:
                            ttl.copy(pipe, recv_blk[0:1, 0:1]).wait()
                        if node_x == 2:
                            ttl.copy(pipe, recv_blk[0:1, 1:2]).wait()

                    bcast_net.if_dst(recv)

        @ttl.datamovement()
        def write_output():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 1:
                with recv_dfb.wait() as recv_blk:
                    ttl.copy(recv_blk[0:1, 0:1], out[0, 0]).wait()
            if node_x == 2:
                with recv_dfb.wait() as recv_blk:
                    ttl.copy(recv_blk[0:1, 1:2], out[0, 1]).wait()

    return non_uniform_multicast_receive_address


non_uniform_multicast_receive_address_kernel = (
    make_non_uniform_multicast_receive_address_kernel()
)


def test_posted_gather_uses_distinct_receiver_slots(device):
    inp_torch = torch.randn(TILE, 2 * TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    posted_gather_kernel(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    expected = (
        inp_torch[:, 0:TILE].float() + inp_torch[:, TILE : 2 * TILE].float()
    ).to(torch.bfloat16)
    assert_pcc(expected.float(), result.float())


def test_same_source_pipes_use_distinct_rendezvous_state(device):
    inp_torch = torch.randn(TILE, 2 * TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    same_source_two_pipe_kernel(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    assert_pcc(inp_torch.float(), result.float())


@pytest.mark.xfail(
    reason=(
        "issue #617: multicast lowering assumes all destinations publish the "
        "same receive address"
    ),
    strict=True,
)
def test_multicast_receive_addresses_can_differ_by_destination(device):
    inp_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    non_uniform_multicast_receive_address_kernel(inp, out)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out)
    expected = inp_torch.repeat(1, 2)
    assert_pcc(expected.float(), result.float())
