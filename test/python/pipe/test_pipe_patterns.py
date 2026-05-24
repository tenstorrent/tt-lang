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
# Scatter-gather (1D): single row, each core multicasts to all cores
# (all-to-all along x).
# ---------------------------------------------------------------------------

N_SG = 4


@ttl.operation(grid=(N_SG, 1))
def scatter_gather_1d_kernel(inp, out):
    # All-to-all on one row: each core multicasts to all (loopback included).
    net = ttl.PipeNet(
        [ttl.Pipe(src=(x, 0), dst=(slice(0, N_SG), 0)) for x in range(N_SG)]
    )

    send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    # block_count == N_SG so each sender lands in a distinct block.
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=N_SG)
    acc_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with recv_cb.wait() as recv_blk, acc_cb.reserve() as acc_blk:
            acc_blk.store(recv_blk)
        for _ in range(N_SG - 1):
            with (
                recv_cb.wait() as recv_blk,
                acc_cb.wait() as prev_blk,
                acc_cb.reserve() as acc_blk,
            ):
                acc_blk.store(prev_blk + recv_blk)
        with acc_cb.wait() as acc_blk, out_cb.reserve() as out_blk:
            out_blk.store(acc_blk)

    # Sender and receiver run on separate NOC threads. Combining them on
    # one thread deadlocks: every core blocks on its own sender handshake
    # (waiting for receivers to signal ready) before any if_dst block can
    # run to issue those signals.
    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        with send_cb.reserve() as send_blk:
            ttl.copy(inp[0, x], send_blk).wait()

            def send(pipe):
                ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)

        def recv(pipe):
            with recv_cb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        net.if_dst(recv)

        with out_cb.wait() as out_blk:
            ttl.copy(out_blk, out[0, x]).wait()


# ---------------------------------------------------------------------------
# Scatter-gather (2D): per-column all-to-all (matches TTLangSpecification.md
# scatter-gather example). Each column x independently performs an
# all-to-all across its SY_SG rows; loopback included via slice(0, SY_SG).
# ---------------------------------------------------------------------------

SX_SG = 2
SY_SG = 3


@ttl.operation(grid=(SX_SG, SY_SG))
def scatter_gather_2d_kernel(inp, out):
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(x, y), dst=(x, slice(0, SY_SG)))
            for x in range(SX_SG)
            for y in range(SY_SG)
        ]
    )

    send_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    # block_count == SY_SG so each sender on a column lands in a distinct block.
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=SY_SG)
    acc_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with recv_cb.wait() as recv_blk, acc_cb.reserve() as acc_blk:
            acc_blk.store(recv_blk)
        for _ in range(SY_SG - 1):
            with (
                recv_cb.wait() as recv_blk,
                acc_cb.wait() as prev_blk,
                acc_cb.reserve() as acc_blk,
            ):
                acc_blk.store(prev_blk + recv_blk)
        with acc_cb.wait() as acc_blk, out_cb.reserve() as out_blk:
            out_blk.store(acc_blk)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with send_cb.reserve() as send_blk:
            ttl.copy(inp[y, x], send_blk).wait()

            def send(pipe):
                ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)

        def recv(pipe):
            with recv_cb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()

        net.if_dst(recv)

        with out_cb.wait() as out_blk:
            ttl.copy(out_blk, out[y, x]).wait()


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


def test_scatter_gather_1d(device):
    """1D all-to-all (issue #505): each core multicasts its tile to all cores
    via one PipeNet; each receiver sums the N_SG tiles."""
    inp_torch = torch.randn(TILE, N_SG * TILE, dtype=torch.bfloat16) * 0.1

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, N_SG * TILE, dtype=torch.bfloat16), device)

    scatter_gather_1d_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    # Each core receives all 4 tiles and sums them
    total = sum(
        inp_torch[:, x * TILE : (x + 1) * TILE].float() for x in range(N_SG)
    ).to(torch.bfloat16)
    expected = total.repeat(1, N_SG)
    assert_pcc(expected, result)


def test_scatter_gather_2d(device):
    """2D per-column all-to-all matching the spec scatter-gather example.
    Column x performs an independent all-to-all across its SY_SG rows; each
    core (x, y) sums the SY_SG tiles in column x and writes the sum back to
    (x, y). Output rows of the same column are identical."""
    inp_torch = torch.randn(SY_SG * TILE, SX_SG * TILE, dtype=torch.bfloat16) * 0.1
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(
        torch.zeros(SY_SG * TILE, SX_SG * TILE, dtype=torch.bfloat16), device
    )

    scatter_gather_2d_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    col_sums = [
        sum(
            inp_torch[y * TILE : (y + 1) * TILE, x * TILE : (x + 1) * TILE].float()
            for y in range(SY_SG)
        ).to(torch.bfloat16)
        for x in range(SX_SG)
    ]
    expected_row = torch.cat(col_sums, dim=1)
    expected = expected_row.repeat(SY_SG, 1)
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
# Per-row forward rings: a USE_Y x USE_X subgrid of the launched device grid
# forms USE_Y independent rings. Every active core (x, y) sends its tile to
# ((x+1) % USE_X, y) and adds its predecessor's tile to its own. The ring
# extent is bounded so `if_src` / `if_dst` per-pipe expansions stay within
# NCRISC code memory on small-code-region devices (e.g. Wormhole, 0x4000).
# Nodes outside the active subgrid skip every body via `if net.is_active()`.
# ---------------------------------------------------------------------------

RING_X = 4
RING_Y = 4


@ttl.operation(grid="full")
def row_rings_kernel(inp, out):
    grid_x, grid_y = ttl.grid_size(dims=2)
    use_x = min(grid_x, RING_X)
    use_y = min(grid_y, RING_Y)
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(x, y), dst=((x + 1) % use_x, y))
            for y in range(use_y)
            for x in range(use_x)
        ]
    )

    own_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    nbr_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if net.is_active():
            with own_cb.wait() as own, nbr_cb.wait() as nbr, out_cb.reserve() as o:
                o.store(own + nbr)

    @ttl.datamovement()
    def dm_read():
        if net.is_active():
            x, y = ttl.node(dims=2)
            with own_cb.reserve() as blk:
                ttl.copy(inp[y, x], blk).wait()

                def send(pipe):
                    ttl.copy(blk, pipe).wait()

                net.if_src(send)

            with nbr_cb.reserve() as blk:

                def recv(pipe):
                    ttl.copy(pipe, blk).wait()

                net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        if net.is_active():
            x, y = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[y, x]).wait()


def test_row_rings_full(device):
    """USE_Y parallel forward rings on a USE_Y x USE_X subgrid of the device
    compute grid (USE_X = min(grid_x, RING_X), USE_Y = min(grid_y, RING_Y)).
    Each active receiver computes
    out[y, x] = inp[y, x] + inp[y, (x-1) % USE_X]."""
    grid = device.compute_with_storage_grid_size()
    use_x = min(grid.x, RING_X)
    use_y = min(grid.y, RING_Y)

    inp_torch = torch.randn(use_y * TILE, use_x * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(
        torch.zeros(use_y * TILE, use_x * TILE, dtype=torch.bfloat16), device
    )

    row_rings_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected = torch.zeros_like(inp_torch)
    for y in range(use_y):
        for x in range(use_x):
            own = inp_torch[y * TILE : (y + 1) * TILE, x * TILE : (x + 1) * TILE]
            prev = (x - 1) % use_x
            nbr = inp_torch[y * TILE : (y + 1) * TILE, prev * TILE : (prev + 1) * TILE]
            expected[y * TILE : (y + 1) * TILE, x * TILE : (x + 1) * TILE] = own + nbr
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
