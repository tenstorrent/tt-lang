# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipe protocol equivalence and computed-address schedule regressions.

The point-to-point pipe is eligible for computed receiver addresses by default;
--no-ttl-pipe-capacity-sync selects receiver-post synchronization, and
--no-ttl-pipe-computed-addresses selects receiver-published addresses. All
three protocols must match torch and each other across dtypes. A receiver block
count greater than one exercises dynamic sender-side slot indices. The schedule
regressions require receiver-published addresses when dynamic control flow or a
repeated PipeKey prevents static receiver-slot assignment. Invalid schedule
regressions reject receiver DFB divergence and unequal dynamic rendezvous
counts.
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


def _a_twice_then_b(inp, out):
    pipe_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(2, 0))])
    pipe_b = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(2, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for _iter_idx in range(2):

            def send_a(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            pipe_a.if_src(send_a)

            def recv_a(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            pipe_a.if_dst(recv_a)

        def send_b(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 1], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        pipe_b.if_src(send_b)

        def recv_b(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()

        pipe_b.if_dst(recv_b)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(3, 1))
def loop_multiplicity_pipe(inp, out):
    _a_twice_then_b(inp, out)


def _branch_ordered_transfers(inp, out):
    pipe_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(2, 0))])
    pipe_b = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(2, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for iter_idx in range(2):
            if iter_idx > 0:

                def send_a(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                pipe_a.if_src(send_a)

                def recv_a(pipe):
                    with recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                pipe_a.if_dst(recv_a)
            else:

                def send_b(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                pipe_b.if_src(send_b)

                def recv_b(pipe):
                    with recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                pipe_b.if_dst(recv_b)

        def drain_receiver(_pipe):
            with recv_dfb.wait() as first_blk:
                ttl.copy(first_blk, out[0, 0]).wait()
            with recv_dfb.wait() as second_blk:
                ttl.copy(second_blk, out[0, 1]).wait()

        pipe_a.if_dst(drain_receiver)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(3, 1))
def branch_order_pipe(inp, out):
    _branch_ordered_transfers(inp, out)


def _same_pipe_key_a_a_b(inp, out):
    pipe_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, 0))])
    pipe_b = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        def transfer_a_twice(pipe):
            with recv_dfb.reserve() as recv_blk:
                first_receive = ttl.copy(pipe, recv_blk)
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()
                first_receive.wait()

            with recv_dfb.reserve() as recv_blk:
                second_receive = ttl.copy(pipe, recv_blk)
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 1], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()
                second_receive.wait()

            with recv_dfb.wait() as _discarded_blk:
                pass

        pipe_a.if_dst(transfer_a_twice)

        def receive_b_then_read_oldest(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as oldest_blk:
                ttl.copy(oldest_blk, out[0, 0]).wait()

        pipe_b.if_dst(receive_b_then_read_oldest)

        def send_b(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 2], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        pipe_b.if_src(send_b)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def repeated_pipe_key(inp, out):
    _same_pipe_key_a_a_b(inp, out)


def _multicast_with_asymmetric_receiver_traffic(inp, out):
    collective = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))])
    receiver_one = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        def advance_receiver_one(_pipe):
            with recv_dfb.reserve() as local_blk:
                ttl.copy(inp[0, 0], local_blk).wait()
            with recv_dfb.wait() as _discarded_blk:
                pass

        receiver_one.if_dst(advance_receiver_one)

        def send(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        collective.if_src(send)

        def recv(pipe):
            node_x, _node_y = ttl.node(dims=2)
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, node_x - 1]).wait()

        collective.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(3, 1))
def asymmetric_multicast_receiver_traffic(inp, out):
    _multicast_with_asymmetric_receiver_traffic(inp, out)


def _send_twice_receive_once(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as _discarded_blk:
                pass

        net.if_dst(recv)

        for _iteration in range(2):

            def send(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def mismatched_pipe_occurrences(inp, out):
    _send_twice_receive_once(inp, out)


def _node_dependent_rendezvous_count(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        for _iteration in range(node_x + 1):

            def send(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

            def receive(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            net.if_dst(receive)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def node_dependent_rendezvous_count(inp, out):
    _node_dependent_rendezvous_count(inp, out)


def _loop_conditional_rendezvous_count(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for iteration in range(2):
            if iteration == 0:

                def send(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                net.if_src(send)

            def receive(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            net.if_dst(receive)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def loop_conditional_rendezvous_count(inp, out):
    _loop_conditional_rendezvous_count(inp, out)


def _random_tiles(count, dtype):
    generator = torch.Generator().manual_seed(0)
    return torch.randn((TILE, count * TILE), generator=generator).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_loop_multiplicity_uses_published_addresses(device, dtype):
    inp_torch = _random_tiles(2, dtype)
    output = to_dram(torch.zeros((TILE, TILE), dtype=dtype), device)

    loop_multiplicity_pipe(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    assert_pcc(inp_torch[:, TILE:].float(), ttnn.to_torch(output).float())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_branch_order_uses_published_addresses(device, dtype):
    inp_torch = _random_tiles(2, dtype)
    output = to_dram(torch.zeros_like(inp_torch), device)

    branch_order_pipe(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    expected = torch.cat((inp_torch[:, TILE:], inp_torch[:, :TILE]), dim=1)
    assert_pcc(expected.float(), ttnn.to_torch(output).float())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_repeated_pipe_key_uses_published_addresses(device, dtype):
    inp_torch = _random_tiles(3, dtype)
    output = to_dram(torch.zeros((TILE, TILE), dtype=dtype), device)

    repeated_pipe_key(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    assert_pcc(inp_torch[:, TILE : 2 * TILE].float(), ttnn.to_torch(output).float())


def test_multicast_rejects_asymmetric_receiver_dfb_traffic(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, 2 * TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "collective pipe receiver DFB write pointers are not proven equal; "
            "TT-Metal NoC multicast requires one destination SRAM address for "
            "all receivers"
        ),
    ):
        asymmetric_multicast_receiver_traffic(to_dram(inp_torch, device), output)


def test_pipe_rejects_different_rendezvous_execution_contexts(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "cannot prove a one-to-one synchronization schedule on PipeNet.*"
            "receiver post and send occurrences do not have matching proven "
            "execution counts and conditions"
        ),
    ):
        mismatched_pipe_occurrences(to_dram(inp_torch, device), output)


def test_pipe_rejects_node_dependent_rendezvous_count(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "cannot prove a one-to-one synchronization schedule on PipeNet.*"
            "receiver post and send occurrences do not have matching proven "
            "execution counts and conditions"
        ),
    ):
        node_dependent_rendezvous_count(to_dram(inp_torch, device), output)


def test_pipe_rejects_loop_conditional_rendezvous_count(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "cannot prove a one-to-one synchronization schedule on PipeNet.*"
            "receiver post and send occurrences do not have matching proven "
            "execution counts and conditions"
        ),
    ):
        loop_conditional_rendezvous_count(to_dram(inp_torch, device), output)
