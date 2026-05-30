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


def make_many_pipe_rendezvous_kernel():
    grid_dim = 2
    row_upper_net = ttl.PipeNet(
        [
            ttl.Pipe((0, row_idx), (slice(row_idx, grid_dim), row_idx))
            for row_idx in range(grid_dim)
        ]
    )
    row_lower_net = ttl.PipeNet(
        [
            ttl.Pipe((0, row_idx), (slice(0, row_idx), row_idx))
            for row_idx in range(1, grid_dim)
        ]
    )
    col_upper_net = ttl.PipeNet(
        [
            ttl.Pipe(
                (col_idx, 0),
                (col_idx, slice(0, col_idx + 1)),
            )
            for col_idx in range(grid_dim)
        ]
    )
    col_lower_net = ttl.PipeNet(
        [
            ttl.Pipe(
                (col_idx, 0),
                (col_idx, slice(col_idx + 1, grid_dim)),
            )
            for col_idx in range(0, grid_dim - 1)
        ]
    )
    helper_row_even_net = ttl.PipeNet(
        [ttl.Pipe((0, row_idx), (grid_dim, row_idx)) for row_idx in range(grid_dim)]
    )
    helper_col_even_net = ttl.PipeNet(
        [ttl.Pipe((row_idx, 0), (grid_dim, row_idx)) for row_idx in range(grid_dim)]
    )

    @ttl.operation(grid=(grid_dim + 1, grid_dim), fp32_dest_acc_en=True)
    def many_pipe_rendezvous(inp, out):
        _row_upper_net = row_upper_net
        _row_lower_net = row_lower_net
        _col_upper_net = col_upper_net
        _col_lower_net = col_lower_net
        _helper_row_even_net = helper_row_even_net
        _helper_col_even_net = helper_col_even_net

        half_k = inp.shape[1] // (2 * TILE)
        tile11 = (1, 1)
        row_recv_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=tile11, block_count=half_k
        )
        col_recv_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=tile11, block_count=half_k
        )
        row_send_dfb = ttl.make_dataflow_buffer_like(inp, shape=tile11, block_count=2)
        col_send_dfb = ttl.make_dataflow_buffer_like(inp, shape=tile11, block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def post_receives_and_send():
            node_x, node_y = ttl.node(dims=2)
            for k_pair in range(half_k):
                even_k = 2 * k_pair
                odd_k = even_k + 1

                def recv_row(pipe):
                    ttl.copy(pipe, row_recv_blk).wait()

                def recv_col(pipe):
                    ttl.copy(pipe, col_recv_blk).wait()

                if row_lower_net.is_src():
                    with row_send_dfb.reserve() as row_send_blk:
                        ttl.copy(
                            inp[node_y : node_y + 1, even_k : even_k + 1], row_send_blk
                        ).wait()

                        def send_row(pipe):
                            ttl.copy(row_send_blk, pipe).wait()

                        if row_lower_net.is_dst():
                            with row_recv_dfb.reserve() as row_recv_blk:

                                def recv_row_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, row_recv_blk)
                                    row_lower_net.if_src(send_row)
                                    helper_row_even_net.if_src(send_row)
                                    recv_tx.wait()

                                row_lower_net.if_dst(recv_row_then_send)
                        else:
                            row_lower_net.if_src(send_row)
                            helper_row_even_net.if_src(send_row)
                elif helper_row_even_net.is_src():
                    with row_send_dfb.reserve() as row_send_blk:
                        ttl.copy(
                            inp[node_y : node_y + 1, even_k : even_k + 1], row_send_blk
                        ).wait()

                        def send_row(pipe):
                            ttl.copy(row_send_blk, pipe).wait()

                        helper_row_even_net.if_src(send_row)
                elif row_lower_net.is_dst():
                    with row_recv_dfb.reserve() as row_recv_blk:
                        row_lower_net.if_dst(recv_row)
                elif helper_row_even_net.is_dst():
                    with row_recv_dfb.reserve() as row_recv_blk:
                        helper_row_even_net.if_dst(recv_row)

                if col_lower_net.is_src():
                    with col_send_dfb.reserve() as col_send_blk:
                        ttl.copy(
                            inp[node_x : node_x + 1, even_k : even_k + 1], col_send_blk
                        ).wait()

                        def send_col(pipe):
                            ttl.copy(col_send_blk, pipe).wait()

                        if col_lower_net.is_dst():
                            with col_recv_dfb.reserve() as col_recv_blk:

                                def recv_col_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, col_recv_blk)
                                    col_lower_net.if_src(send_col)
                                    helper_col_even_net.if_src(send_col)
                                    recv_tx.wait()

                                col_lower_net.if_dst(recv_col_then_send)
                        else:
                            col_lower_net.if_src(send_col)
                            helper_col_even_net.if_src(send_col)
                elif helper_col_even_net.is_src():
                    with col_send_dfb.reserve() as col_send_blk:
                        ttl.copy(
                            inp[node_x : node_x + 1, even_k : even_k + 1], col_send_blk
                        ).wait()

                        def send_col(pipe):
                            ttl.copy(col_send_blk, pipe).wait()

                        helper_col_even_net.if_src(send_col)
                elif col_lower_net.is_dst():
                    with col_recv_dfb.reserve() as col_recv_blk:
                        col_lower_net.if_dst(recv_col)
                elif helper_col_even_net.is_dst():
                    with col_recv_dfb.reserve() as col_recv_blk:
                        helper_col_even_net.if_dst(recv_col)

                if row_upper_net.is_src():
                    with row_send_dfb.reserve() as row_send_blk:
                        ttl.copy(
                            inp[node_y : node_y + 1, odd_k : odd_k + 1], row_send_blk
                        ).wait()

                        def send_row(pipe):
                            ttl.copy(row_send_blk, pipe).wait()

                        if row_upper_net.is_dst():
                            with row_recv_dfb.reserve() as row_recv_blk:

                                def recv_row_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, row_recv_blk)
                                    row_upper_net.if_src(send_row)
                                    recv_tx.wait()

                                row_upper_net.if_dst(recv_row_then_send)
                        else:
                            row_upper_net.if_src(send_row)
                elif row_upper_net.is_dst():
                    with row_recv_dfb.reserve() as row_recv_blk:
                        row_upper_net.if_dst(recv_row)

                if col_upper_net.is_src():
                    with col_send_dfb.reserve() as col_send_blk:
                        ttl.copy(
                            inp[node_x : node_x + 1, odd_k : odd_k + 1], col_send_blk
                        ).wait()

                        def send_col(pipe):
                            ttl.copy(col_send_blk, pipe).wait()

                        if col_upper_net.is_dst():
                            with col_recv_dfb.reserve() as col_recv_blk:

                                def recv_col_then_send(pipe):
                                    recv_tx = ttl.copy(pipe, col_recv_blk)
                                    col_upper_net.if_src(send_col)
                                    recv_tx.wait()

                                col_upper_net.if_dst(recv_col_then_send)
                        else:
                            col_upper_net.if_src(send_col)
                elif col_upper_net.is_dst():
                    with col_recv_dfb.reserve() as col_recv_blk:
                        col_upper_net.if_dst(recv_col)

        @ttl.datamovement()
        def write_output():
            pass

    return many_pipe_rendezvous


def make_non_uniform_multicast_receive_address_kernel():
    bcast_pipe = ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))
    bcast_net = ttl.PipeNet([bcast_pipe])

    @ttl.operation(grid=(3, 1))
    def non_uniform_multicast_receive_address(inp, out):
        _bcast_net = bcast_net

        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        first_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        second_recv_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )

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
            if node_x == 1:
                with first_recv_dfb.reserve() as recv_blk:
                    ttl.copy(bcast_pipe, recv_blk).wait()
            if node_x == 2:
                with second_recv_dfb.reserve() as recv_blk:
                    ttl.copy(bcast_pipe, recv_blk).wait()

        @ttl.datamovement()
        def write_output():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 1:
                with first_recv_dfb.wait() as recv_blk:
                    ttl.copy(recv_blk, out[0, 0]).wait()
            if node_x == 2:
                with second_recv_dfb.wait() as recv_blk:
                    ttl.copy(recv_blk, out[0, 1]).wait()

    return non_uniform_multicast_receive_address


non_uniform_multicast_receive_address_kernel = (
    make_non_uniform_multicast_receive_address_kernel()
)
many_pipe_rendezvous_kernel = make_many_pipe_rendezvous_kernel()


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


def test_many_pipe_rendezvous_sites_report_hardware_semaphore_limit(device):
    inp_torch = torch.randn(2 * TILE, 2 * TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(2 * TILE, 2 * TILE, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    with pytest.raises(
        Exception,
        match=(
            "pipe rendezvous requires .* hardware semaphore ids, exceeding "
            "TT hardware limit of 16; issue #619"
        ),
    ):
        many_pipe_rendezvous_kernel(inp, out)


def test_multicast_receive_addresses_differ_by_destination_rejected(device):
    inp_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    with pytest.raises(
        Exception,
        match=(
            "multicast pipe receive posts publish non-uniform destination "
            "addresses; per-destination multicast receive addresses are "
            "tracked by issue #617"
        ),
    ):
        non_uniform_multicast_receive_address_kernel(inp, out)
