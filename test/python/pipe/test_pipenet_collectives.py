# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device pytests for collective patterns under grid="auto" with launch
extent larger than work extent.

`test/python/pipe/test_pipe_patterns.py` already covers the basic
gather, scatter, scatter-gather, and ring forward kernels with launch
extent equal to work extent. The cases here cover regimes the
`ttl-verify-pipenet-guards` verifier exercises under `grid="auto"`:

* Scatter on a subgrid (`grid="auto"`, work = 4 nodes in row 0):
  single PipeNet, single multicast pipe, dst rectangle smaller than the
  launch grid.
* Per-row scatter (`grid="auto"`, work = ROWS x COLS): single PipeNet
  with multiple pipes whose destination rectangles do not overlap
  (different rows). 2D active set.
* Two PipeNets with overlapping destinations: a single cross-PipeNet
  overlap is permitted (the within-PipeNet rule from issue #505 only
  rejects overlap inside one PipeNet).

True all-to-all and ring all-reduce in a *single* PipeNet are blocked on
issue #505 (within-PipeNet multicast destination overlap). The
per-source PipeNet workaround is sketched in TODO comments.
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
# Scatter on a subgrid: node (0, 0) multicasts a tile to nodes 1..N-1 in
# row 0. Single PipeNet, single multicast pipe.
# ---------------------------------------------------------------------------


N_SCATTER = 4


@ttl.operation(grid="auto")
def scatter_subgrid_kernel(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, N_SCATTER), 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if net.is_active():
            with inp_cb.wait() as t, out_cb.reserve() as o:
                o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
        if net.is_active():
            with inp_cb.reserve() as blk:

                def read_and_send(pipe):
                    ttl.copy(inp[0, 0], blk).wait()
                    ttl.copy(blk, pipe).wait()

                net.if_src(read_and_send)

                def recv(pipe):
                    ttl.copy(pipe, blk).wait()

                net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        if net.is_active():
            x, _ = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, x]).wait()


def test_scatter_subgrid(device):
    """Scatter from (0, 0) to (slice(1, 4), 0) under grid="auto".

    Active set: {(0,0), (1,0), (2,0), (3,0)}. The launch extent equals
    the active set, so every launched node carries a PipeNet role and
    the verifier accepts the unguarded `if_src` / `if_dst` bodies.
    """
    inp_torch = torch.randn(TILE, N_SCATTER * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, N_SCATTER * TILE, dtype=torch.bfloat16), device)

    scatter_subgrid_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    tile0 = torch.abs(inp_torch[:, 0:TILE])
    expected = tile0.repeat(1, N_SCATTER)
    assert_pcc(expected, result)


# ---------------------------------------------------------------------------
# Per-row scatter on a subgrid: single PipeNet, ROWS multicast pipes whose
# destination rectangles are disjoint (different rows). Each row r
# multicasts inp's r-th tile from (0, r) to (slice(1, COLS), r).
# Source nodes (0, r) consume their own tile via dm_read directly.
# ---------------------------------------------------------------------------


PR_ROWS = 3
PR_COLS = 4


@ttl.operation(grid="auto")
def per_row_scatter_kernel(inp, out):
    net = ttl.PipeNet(
        [ttl.Pipe(src=(0, r), dst=(slice(1, PR_COLS), r)) for r in range(PR_ROWS)]
    )

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if net.is_active():
            with inp_cb.wait() as t, out_cb.reserve() as o:
                o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
        if net.is_active():
            x, y = ttl.node(dims=2)
            with inp_cb.reserve() as blk:

                def read_and_send(pipe):
                    ttl.copy(inp[y, 0], blk).wait()
                    ttl.copy(blk, pipe).wait()

                net.if_src(read_and_send)

                def recv(pipe):
                    ttl.copy(pipe, blk).wait()

                net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        if net.is_active():
            x, y = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[y, x]).wait()


def test_per_row_scatter(device):
    """ROWS independent scatters in one PipeNet, dst rectangles disjoint.

    Active set is the ROWS x COLS rectangle; nodes beyond skip the body.
    Each output row r holds abs(inp[r, 0]) tiled across COLS columns.
    """
    inp_torch = torch.randn(PR_ROWS * TILE, PR_COLS * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(
        torch.zeros(PR_ROWS * TILE, PR_COLS * TILE, dtype=torch.bfloat16), device
    )

    per_row_scatter_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected = torch.empty_like(inp_torch)
    for r in range(PR_ROWS):
        tile0 = torch.abs(inp_torch[r * TILE : (r + 1) * TILE, 0:TILE])
        for c in range(PR_COLS):
            expected[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = tile0
    assert_pcc(expected, result)


# ---------------------------------------------------------------------------
# Two PipeNets whose destination rectangles intersect.
#
#   net_a: src=(0,0) -> dst=slice(1,3),0     (multicast to nodes 1 and 2)
#   net_b: src=(3,0) -> dst=slice(1,3),0     (multicast to nodes 1 and 2)
#
# Nodes 1 and 2 are destinations of both PipeNets. The within-PipeNet
# overlap rule (issue #505) only forbids overlap inside a single
# PipeNet; cross-PipeNet overlap is allowed because each PipeNet has
# its own semaphore pair. Each receiver sums the two tiles it gets.
# ---------------------------------------------------------------------------


@ttl.operation(grid="auto")
def overlapping_pipenets_kernel(inp, out):
    net_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))])
    net_b = ttl.PipeNet([ttl.Pipe(src=(3, 0), dst=(slice(1, 3), 0))])

    a_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    b_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        # Only nodes 1 and 2 receive both inputs and produce output.
        if net_a.is_active() or net_b.is_active():
            x, _ = ttl.node(dims=2)
            if 1 <= x and x <= 2:
                with a_cb.wait() as a, b_cb.wait() as b, out_cb.reserve() as o:
                    o.store(a + b)

    @ttl.datamovement()
    def dm_read():
        # Reserve only the CBs that this node will actually write or read.
        # The simulator (correctly) rejects a reserved CB that exits its
        # `with` block in the MW state without ever being written; hardware
        # is more permissive but the conditional structure is still valid.
        if net_a.is_active() or net_b.is_active():
            x, _ = ttl.node(dims=2)
            if x == 0:
                # net_a source.
                with a_cb.reserve() as ablk:

                    def src_a(pipe):
                        ttl.copy(inp[0, 0], ablk).wait()
                        ttl.copy(ablk, pipe).wait()

                    net_a.if_src(src_a)
            elif x == 3:
                # net_b source.
                with b_cb.reserve() as bblk:

                    def src_b(pipe):
                        ttl.copy(inp[0, 3], bblk).wait()
                        ttl.copy(bblk, pipe).wait()

                    net_b.if_src(src_b)
            elif 1 <= x and x <= 2:
                # Destination of both: receive from each net into its own CB.
                with a_cb.reserve() as ablk:

                    def dst_a(pipe):
                        ttl.copy(pipe, ablk).wait()

                    net_a.if_dst(dst_a)
                with b_cb.reserve() as bblk:

                    def dst_b(pipe):
                        ttl.copy(pipe, bblk).wait()

                    net_b.if_dst(dst_b)

    @ttl.datamovement()
    def dm_write():
        if net_a.is_active() or net_b.is_active():
            x, _ = ttl.node(dims=2)
            if 1 <= x and x <= 2:
                with out_cb.wait() as blk:
                    ttl.copy(blk, out[0, x]).wait()


def test_overlapping_pipenets(device):
    """Two scatters whose dst rectangles share nodes 1 and 2.

    Active set is the union {(0,0), (1,0), (2,0), (3,0)}. Nodes 1 and 2
    receive from both PipeNets and sum the two tiles. Nodes 0 and 3 are
    pure sources: the compute and dm_write bodies guard with
    `1 <= x <= 2`, so they don't run compute or write to `out`. Only
    assert on columns 1 and 2.

    TODO[spec]: the spec does not constrain cross-PipeNet destination
    overlap; the implementation supports it because each PipeNet
    allocates its own semaphore pair.
    """
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16) * 0.1
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 4 * TILE, dtype=torch.bfloat16), device)

    overlapping_pipenets_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)

    # Nodes 1 and 2 should hold inp[:, 0:TILE] + inp[:, 3*TILE:4*TILE].
    expected_mid = (
        inp_torch[:, 0:TILE].float() + inp_torch[:, 3 * TILE : 4 * TILE].float()
    )
    for col in (1, 2):
        actual = result[:, col * TILE : (col + 1) * TILE].float()
        # bfloat16 addition; loose tolerance.
        diff = (expected_mid - actual).abs().max().item()
        assert diff < 0.05, (
            f"node {col} mismatch: max diff {diff}, "
            f"expected={expected_mid[:1, :4]}, actual={actual[:1, :4]}"
        )


# ---------------------------------------------------------------------------
# Nested if_src/if_dst across two PipeNets, per spec line 645:
# "Calls into if_src and if_dst can be nested within condition functions
#  for different pipe nets."
#
#   net_a: unicast (1, 0) -> (0, 0)
#   net_b: unicast (0, 0) -> (2, 0)
#
# The dm_read on node 0 receives via net_a and, inside that callback,
# forwards via net_b — exercising the nesting form.
# ---------------------------------------------------------------------------


@ttl.operation(grid="auto")
def nested_if_callbacks_kernel(inp, out):
    net_a = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    net_b = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(2, 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if net_a.is_active() or net_b.is_active():
            with inp_cb.wait() as t, out_cb.reserve() as o:
                o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
        if net_a.is_active() or net_b.is_active():
            with inp_cb.reserve() as blk:

                # Node 1: source for net_a — read its tile and send.
                def send_a(pipe_a):
                    ttl.copy(inp[0, 1], blk).wait()
                    ttl.copy(blk, pipe_a).wait()

                net_a.if_src(send_a)

                # Node 0: receives from net_a, then in the SAME callback acts as
                # net_b source and forwards. The spec permits this nesting for
                # different pipe nets.
                def recv_a_then_send_b(pipe_a):
                    ttl.copy(pipe_a, blk).wait()

                    def send_b(pipe_b):
                        ttl.copy(blk, pipe_b).wait()

                    net_b.if_src(send_b)

                net_a.if_dst(recv_a_then_send_b)

                # Node 2: net_b destination — receive.
                def recv_b(pipe_b):
                    ttl.copy(pipe_b, blk).wait()

                net_b.if_dst(recv_b)

    @ttl.datamovement()
    def dm_write():
        if net_a.is_active() or net_b.is_active():
            x, _ = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, x]).wait()


# ---------------------------------------------------------------------------
# Loopback multicast: source (0, 0) is inside its own destination range
# (column 0, rows 0..3). Each of the 4 nodes in column 0 receives the
# tile and writes it; the source receives its own data through the
# multicast handshake, so the kernel does not need a separate
# read-first branch on it.
# ---------------------------------------------------------------------------


N_LB = 4


@ttl.operation(grid="auto")
def loopback_multicast_kernel(inp, out):
    # src=(0,0); dst column 0 rows 0..3 (includes source).
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, slice(0, N_LB)))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if net.is_active():
            with inp_cb.wait() as t, out_cb.reserve() as o:
                o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
        if net.is_active():
            with inp_cb.reserve() as blk:

                def src(pipe):
                    ttl.copy(inp[0, 0], blk).wait()
                    ttl.copy(blk, pipe).wait()

                net.if_src(src)

                def dst(pipe):
                    ttl.copy(pipe, blk).wait()

                net.if_dst(dst)

    @ttl.datamovement()
    def dm_write():
        if net.is_active():
            _, y = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[y, 0]).wait()


def test_loopback_multicast(device):
    """Source-in-destination-range multicast (loopback).

    The source is included in its own destination range, so all N_LB
    destinations (including the source) end up holding the same tile.
    Verifies the source receives via the multicast handshake rather
    than retaining whatever it locally read.

    TODO[spec]: the spec does not address loopback multicast (source
    inside its own destination range).
    """
    inp_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(N_LB * TILE, TILE, dtype=torch.bfloat16), device)

    loopback_multicast_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    src_tile = torch.abs(inp_torch[0:TILE, 0:TILE])
    expected = src_tile.repeat(N_LB, 1)
    assert_pcc(expected, result)


def test_nested_if_callbacks(device):
    """net_a.if_dst contains a nested net_b.if_src for relay forwarding.

    Spec line 645 explicitly allows this nesting across different
    PipeNets. Active set is {(0,0), (1,0), (2,0)}; the relay route is
    node 1 -> node 0 -> node 2.
    """
    inp_torch = torch.randn(TILE, 3 * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 3 * TILE, dtype=torch.bfloat16), device)

    nested_if_callbacks_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    # Node 0 received inp[:, 1*TILE:2*TILE] (forwarded from node 1).
    # Node 1's compute output was never produced (it had no inp_cb push);
    # so we only assert on node 0 and node 2.
    tile1 = torch.abs(inp_torch[:, TILE : 2 * TILE])
    actual_c0 = result[:, 0:TILE].float()
    actual_c2 = result[:, 2 * TILE : 3 * TILE].float()
    diff_c0 = (tile1.float() - actual_c0).abs().max().item()
    diff_c2 = (tile1.float() - actual_c2).abs().max().item()
    assert diff_c0 < 0.05, f"node 0 (net_a dst) diff: {diff_c0}"
    assert diff_c2 < 0.05, f"node 2 (net_b dst, via relay) diff: {diff_c2}"
