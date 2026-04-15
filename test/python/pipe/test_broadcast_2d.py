# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
2D rectangular multicast: core (0,0) broadcasts to a 2x2 grid of destinations.

Tests the dst=(slice(x0,x1), slice(y0,y1)) pattern where one source
multicasts to a full 2D rectangular region.
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
GRID_X = 2
GRID_Y = 2


@ttl.operation(grid=(GRID_X, GRID_Y))
def broadcast_2d_kernel(inp, out):
    net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, 0),
                dst=(slice(0, GRID_X), slice(0, GRID_Y)),
            )
        ]
    )

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
        x, y = ttl.node(dims=2)
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[y, x])
            tx.wait()


def test_broadcast_2d(device):
    """2D broadcast from (0,0) to 2x2 grid, each core computes abs."""
    inp_torch = torch.randn(GRID_Y * TILE, GRID_X * TILE, dtype=torch.bfloat16)

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(
        torch.zeros(GRID_Y * TILE, GRID_X * TILE, dtype=torch.bfloat16),
        device,
    )

    broadcast_2d_kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    # All cores receive the same tile (0,0) from source and compute abs
    tile00 = torch.abs(inp_torch[0:TILE, 0:TILE])
    expected = tile00.repeat(GRID_Y, GRID_X)
    assert_pcc(expected, result)
