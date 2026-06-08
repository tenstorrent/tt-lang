# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Negative tests for pipe schedules that would deadlock at runtime."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

TILE = 32


@ttl.operation(grid=(2, 1))
def cyclic_forward_kernel(inp):
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(1, 0)),
            ttl.Pipe(src=(1, 0), dst=(0, 0)),
        ]
    )

    send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def send_before_receive():
        x, _ = ttl.node(dims=2)
        with send_cb.reserve() as send_blk:
            ttl.copy(inp[0, x], send_blk).wait()

            def send(pipe):
                ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

        with recv_cb.reserve() as recv_blk:

            def recv(pipe):
                ttl.copy(pipe, recv_blk).wait()

            net.if_dst(recv)


@ttl.operation(grid=(2, 1))
def loopback_wait_before_send_kernel(inp):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(0, 2), 0))])

    send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def wait_on_receive_before_send():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with send_cb.reserve() as send_blk, recv_cb.reserve() as recv_blk:
                ttl.copy(inp[0, 0], send_blk).wait()

                def recv(pipe):
                    recv_tx = ttl.copy(pipe, recv_blk)
                    recv_tx.wait()

                    def send(pipe):
                        ttl.copy(send_blk, pipe).wait()

                    net.if_src(send)

                net.if_dst(recv)
        else:
            with recv_cb.reserve() as recv_blk:

                def recv(pipe):
                    ttl.copy(pipe, recv_blk).wait()

                net.if_dst(recv)


@ttl.operation(grid=(2, 1))
def loopback_send_before_receive_post_kernel(inp):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(0, 2), 0))])

    send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def send_before_local_receive_post():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with send_cb.reserve() as send_blk, recv_cb.reserve() as recv_blk:
                ttl.copy(inp[0, 0], send_blk).wait()

                def send(pipe):
                    ttl.copy(send_blk, pipe).wait()

                net.if_src(send)

                def recv(pipe):
                    ttl.copy(pipe, recv_blk).wait()

                net.if_dst(recv)
        else:
            with recv_cb.reserve() as recv_blk:

                def recv(pipe):
                    ttl.copy(pipe, recv_blk).wait()

                net.if_dst(recv)


@ttl.operation(grid=(2, 1))
def receive_wait_unanalyzable_guard_kernel(inp):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

    send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def receive_wait_under_unsupported_coord_guard():
        node_x, _node_y = ttl.node(dims=2)

        def send(pipe):
            with send_cb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
                ttl.copy(send_blk, pipe).wait()

        net.if_src(send)

        def recv(pipe):
            with recv_cb.reserve() as recv_blk:
                recv_tx = ttl.copy(pipe, recv_blk)
                if (node_x % 2) == 0:
                    recv_tx.wait()

        net.if_dst(recv)


def test_forward_ring_send_before_receive_rejected(device):
    """Same-thread ring send-before-receive is a pipe wait-for cycle."""
    inp_torch = torch.randn(TILE, 2 * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)

    with pytest.raises(Exception, match="pipe schedule contains a wait-for cycle"):
        cyclic_forward_kernel(inp_tt)


def test_loopback_receive_wait_before_send_rejected(device):
    """The receive wait cannot run before the send that completes it."""
    inp_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)

    with pytest.raises(
        Exception,
        match="receive wait occurs before the send that completes it",
    ):
        loopback_wait_before_send_kernel(inp_tt)


def test_loopback_send_before_receive_post_rejected(device):
    """The sender cannot run before the loopback receive publishes its address."""
    inp_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)

    with pytest.raises(
        Exception,
        match="pipe send occurs before the receiver publishes a destination address",
    ):
        loopback_send_before_receive_post_kernel(inp_tt)


def test_receive_wait_unanalyzable_guard_rejected(device):
    """Pipe receive waits under unknown domains must not be omitted."""
    inp_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)

    with pytest.raises(
        Exception,
        match="could not statically analyze the PipeNet guard",
    ):
        receive_wait_unanalyzable_guard_kernel(inp_tt)
