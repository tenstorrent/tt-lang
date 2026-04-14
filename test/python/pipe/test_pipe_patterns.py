# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipe pattern tests matching the four spec examples:
  1. Gather: unicast from multiple sources to one destination
  2. Scatter: multicast from one source to multiple destinations
  3. Scatter-gather: multicast with loopback (all-to-all)
  4. Forward: unicast to +1 neighbor (ring)
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


# ---------------------------------------------------------------------------
# Gather: cores 1-3 send to core 0, which sums via accumulator
# ---------------------------------------------------------------------------

N_GATHER_SOURCES = 3


@ttl.operation(grid=(N_GATHER_SOURCES + 1, 1))
def gather_kernel(inp, out):
    net = ttl.PipeNet(
        [ttl.Pipe(src=(x, 0), dst=(0, 0)) for x in range(1, N_GATHER_SOURCES + 1)]
    )

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(
        inp, shape=(1, 1), block_count=N_GATHER_SOURCES + 1
    )
    acc_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with recv_cb.wait() as t, acc_cb.reserve() as a:
                a.store(t)
            for _ in range(N_GATHER_SOURCES - 1):
                with recv_cb.wait() as t, acc_cb.wait() as prev, acc_cb.reserve() as a:
                    a.store(prev + t)
            with acc_cb.wait() as a, out_cb.reserve() as o:
                o.store(a)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        if x > 0:
            with inp_cb.reserve() as blk:
                tx = ttl.copy(inp[0, x], blk)
                tx.wait()

                def send(pipe):
                    xf = ttl.copy(blk, pipe)
                    xf.wait()

                net.if_src(send)

        def recv(pipe):
            with recv_cb.reserve() as blk:
                xf = ttl.copy(pipe, blk)
                xf.wait()

        net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0])
                tx.wait()


# ---------------------------------------------------------------------------
# Scatter: core 0 multicasts to cores 1-3
# ---------------------------------------------------------------------------


@ttl.operation(grid=(4, 1))
def scatter_kernel(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as tile_in, out_cb.reserve() as tile_out:
            tile_out.store(ttl.math.abs(tile_in))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:

            def read_and_send(pipe):
                tx = ttl.copy(inp[0, 0], blk)
                tx.wait()
                xf = ttl.copy(blk, pipe)
                xf.wait()

            net.if_src(read_and_send)

            def recv(pipe):
                xf = ttl.copy(pipe, blk)
                xf.wait()

            net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, x])
            tx.wait()


# ---------------------------------------------------------------------------
# Scatter-gather: each core multicasts to all cores (all-to-all)
# ---------------------------------------------------------------------------
#
# TODO(#505): blocked on noc_semaphore_inc_multicast support in TTKernel.
# Multicast pipes in a PipeNet share a semaphore pair, so overlapping
# multicast destinations corrupt the handshake. Once #505 lands, this
# kernel can run with a single PipeNet.

N_SG = 4


@ttl.operation(grid=(N_SG, 1))
def scatter_gather_kernel(inp, out):
    grid_x, grid_y = ttl.grid_size(dims=2)

    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(x, y), dst=(x, slice(0, grid_y)))
            for x in range(grid_x)
            for y in range(grid_y)
        ]
    )

    pipe_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    acc_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with pipe_cb.wait() as t, acc_cb.reserve() as a:
            a.store(t)
        for _ in range(N_SG - 1):
            with pipe_cb.wait() as t, acc_cb.wait() as prev, acc_cb.reserve() as a:
                a.store(prev + t)
        with acc_cb.wait() as a, out_cb.reserve() as o:
            o.store(a)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        with pipe_cb.reserve() as blk:

            def pipe_src(pipe):
                ttl.copy(inp[0, x], blk).wait()
                ttl.copy(blk, pipe).wait()

            def pipe_dst(pipe):
                ttl.copy(pipe, blk).wait()

            net.if_src(pipe_src)
            net.if_dst(pipe_dst)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        with out_cb.wait() as blk:
            ttl.copy(blk, out[0, x]).wait()


# ---------------------------------------------------------------------------
# Forward: each core sends to +1 neighbor (ring)
# ---------------------------------------------------------------------------

N_RING = 4


@ttl.operation(grid=(N_RING, 1))
def forward_kernel(inp, out):
    net = ttl.PipeNet(
        [ttl.Pipe(src=(x, 0), dst=((x + 1) % N_RING, 0)) for x in range(N_RING)]
    )

    own_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    nbr_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with own_cb.wait() as own, nbr_cb.wait() as nbr, out_cb.reserve() as o:
            o.store(own + nbr)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        with own_cb.reserve() as blk:
            tx = ttl.copy(inp[0, x], blk)
            tx.wait()

            def send(pipe):
                xf = ttl.copy(blk, pipe)
                xf.wait()

            net.if_src(send)

        with nbr_cb.reserve() as blk:

            def recv(pipe):
                xf = ttl.copy(pipe, blk)
                xf.wait()

            net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, x])
            tx.wait()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_gather(device):
    """Gather: cores 1-3 send to core 0 which sums."""
    inp_torch = torch.randn(TILE, (N_GATHER_SOURCES + 1) * TILE, dtype=torch.bfloat16)

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

    gather_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected = sum(
        inp_torch[:, x * TILE : (x + 1) * TILE].float()
        for x in range(1, N_GATHER_SOURCES + 1)
    ).to(torch.bfloat16)
    assert_pcc(expected, result)


def test_scatter(device):
    """Scatter: core 0 multicasts to cores 1-3, each computes abs."""
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16)

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 4 * TILE, dtype=torch.bfloat16), device)

    scatter_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    tile0 = torch.abs(inp_torch[:, 0:TILE])
    expected = tile0.repeat(1, 4)
    assert_pcc(expected, result)


@pytest.mark.skip(reason="Blocked on #505: overlapping multicast destinations")
def test_scatter_gather(device):
    """Scatter-gather: all-to-all broadcast with loopback, sum received tiles."""
    inp_torch = torch.randn(TILE, N_SG * TILE, dtype=torch.bfloat16) * 0.1

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, N_SG * TILE, dtype=torch.bfloat16), device)

    scatter_gather_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    # Each core receives all 4 tiles and sums them
    total = sum(
        inp_torch[:, x * TILE : (x + 1) * TILE].float() for x in range(N_SG)
    ).to(torch.bfloat16)
    expected = total.repeat(1, N_SG)
    assert_pcc(expected, result)


def test_forward_ring(device):
    """Forward ring: out[x] = inp[x] + inp[(x-1) % N]."""
    inp_torch = torch.randn(TILE, N_RING * TILE, dtype=torch.bfloat16)

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, N_RING * TILE, dtype=torch.bfloat16), device)

    forward_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected = torch.zeros_like(inp_torch)
    for x in range(N_RING):
        own = inp_torch[:, x * TILE : (x + 1) * TILE]
        prev = (x - 1) % N_RING
        nbr = inp_torch[:, prev * TILE : (prev + 1) * TILE]
        expected[:, x * TILE : (x + 1) * TILE] = own + nbr
    assert_pcc(expected, result)


# ---------------------------------------------------------------------------
# Multi-block gather: sources send 1x2 tile blocks to core 0
# Exercises gather slot offsets with cbNumTiles > 1.
# ---------------------------------------------------------------------------

HTILES = 2


N_GATHER_MB_SOURCES = 2


@ttl.operation(grid=(N_GATHER_MB_SOURCES + 1, 1))
def gather_multiblock_kernel(inp, out):
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(1, 0), dst=(0, 0)),
            ttl.Pipe(src=(2, 0), dst=(0, 0)),
        ]
    )

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, HTILES), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(
        inp, shape=(1, HTILES), block_count=N_GATHER_MB_SOURCES + 1
    )
    acc_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, HTILES), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, HTILES), block_count=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with recv_cb.wait() as t, acc_cb.reserve() as a:
                a.store(t)
            for _ in range(N_GATHER_MB_SOURCES - 1):
                with recv_cb.wait() as t, acc_cb.wait() as prev, acc_cb.reserve() as a:
                    a.store(prev + t)
            with acc_cb.wait() as a, out_cb.reserve() as o:
                o.store(a)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        if x > 0:
            with inp_cb.reserve() as blk:
                tx = ttl.copy(inp[0, (x - 1) * HTILES : x * HTILES], blk)
                tx.wait()

                def send(pipe):
                    xf = ttl.copy(blk, pipe)
                    xf.wait()

                net.if_src(send)

        def recv(pipe):
            with recv_cb.reserve() as blk:
                xf = ttl.copy(pipe, blk)
                xf.wait()

        net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0:HTILES])
                tx.wait()


def test_gather_multiblock(device):
    """Gather with multi-tile blocks: 2 sources send 1x2 blocks to core 0."""
    inp_torch = torch.randn(TILE, 2 * HTILES * TILE, dtype=torch.bfloat16)

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, HTILES * TILE, dtype=torch.bfloat16), device)

    gather_multiblock_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    # Core 1 sends inp[:, 0:2*TILE], Core 2 sends inp[:, 2*TILE:4*TILE]
    t0 = inp_torch[:, 0 : HTILES * TILE].float()
    t1 = inp_torch[:, HTILES * TILE : 2 * HTILES * TILE].float()
    expected = (t0 + t1).to(torch.bfloat16)
    assert_pcc(expected, result)
