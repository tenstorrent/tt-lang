# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device pytests for collective patterns under grid="auto" with launch
extent larger than work extent.

`test/python/pipe/test_pipe_patterns.py` already covers the basic
gather, scatter, scatter-gather, and ring forward kernels with launch
extent equal to work extent. The cases here cover the regimes that
exercise `ttl-insert-pipenet-active-guards`:

* Scatter on a subgrid (`grid="auto"`, work = 4 cores in row 0):
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
# Scatter on a subgrid: core (0, 0) multicasts a tile to cores 1..N-1 in
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
        with inp_cb.wait() as t, out_cb.reserve() as o:
            o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
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
        x, _ = ttl.node(dims=2)
        with out_cb.wait() as blk:
            ttl.copy(blk, out[0, x]).wait()


def test_scatter_subgrid(device):
    """Scatter from (0, 0) to (slice(1, 4), 0) under grid="auto".

    Active set: {(0,0), (1,0), (2,0), (3,0)}. Cores outside skip every
    thread body via the inserted scf.if guard.
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
# Source cores (0, r) consume their own tile via dm_read directly.
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
        with inp_cb.wait() as t, out_cb.reserve() as o:
            o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
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
        x, y = ttl.node(dims=2)
        with out_cb.wait() as blk:
            ttl.copy(blk, out[y, x]).wait()


def test_per_row_scatter(device):
    """ROWS independent scatters in one PipeNet, dst rectangles disjoint.

    Active set is the ROWS x COLS rectangle; cores beyond skip the body.
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
# Two PipeNets with overlapping destinations:
#   net_a: src=(0,0) -> dst=slice(1,3),0   (cores 1, 2)
#   net_b: src=(3,0) -> dst=slice(1,3),0   (cores 1, 2)
# Cores 1 and 2 are destinations of both. Within each PipeNet there is
# one pipe with no internal overlap; the cross-PipeNet overlap at cores
# 1 and 2 is permitted and demonstrated working.
# Each receiver sums its two received tiles.
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
        with a_cb.wait() as a, b_cb.wait() as b, out_cb.reserve() as o:
            o.store(a + b)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as ablk:

            def src_a(pipe):
                ttl.copy(inp[0, 0], ablk).wait()
                ttl.copy(ablk, pipe).wait()

            net_a.if_src(src_a)

            def dst_a(pipe):
                ttl.copy(pipe, ablk).wait()

            net_a.if_dst(dst_a)

        with b_cb.reserve() as bblk:

            def src_b(pipe):
                ttl.copy(inp[0, 3], bblk).wait()
                ttl.copy(bblk, pipe).wait()

            net_b.if_src(src_b)

            def dst_b(pipe):
                ttl.copy(pipe, bblk).wait()

            net_b.if_dst(dst_b)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        with out_cb.wait() as blk:
            ttl.copy(blk, out[0, x]).wait()


def test_overlapping_pipenets(device):
    """Two scatters whose dst rectangles share cores 1 and 2.

    Active set is the union {(0,0), (1,0), (2,0), (3,0)}. Cores 1 and 2
    receive from both PipeNets and sum the two tiles. Cores 0 and 3 are
    pure sources; their compute outputs the sum of their own tile and
    whatever the unused other channel held (don't assert on those).
    """
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16) * 0.1
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 4 * TILE, dtype=torch.bfloat16), device)

    overlapping_pipenets_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)

    # Cores 1 and 2 should hold inp[:, 0:TILE] + inp[:, 3*TILE:4*TILE].
    expected_mid = (
        inp_torch[:, 0:TILE].float() + inp_torch[:, 3 * TILE : 4 * TILE].float()
    )
    for col in (1, 2):
        actual = result[:, col * TILE : (col + 1) * TILE].float()
        # bfloat16 addition; loose tolerance.
        diff = (expected_mid - actual).abs().max().item()
        assert diff < 0.05, (
            f"core {col} mismatch: max diff {diff}, "
            f"expected={expected_mid[:1, :4]}, actual={actual[:1, :4]}"
        )
