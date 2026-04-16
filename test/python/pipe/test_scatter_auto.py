# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Scatter with grid="auto" -- the spec scatter example verbatim.

Each column's row-0 core multicasts to all other rows in that column.
Pipe construction uses ttl.grid_size() so it adapts to whatever
device grid grid="auto" resolves to.
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


@ttl.operation(grid="auto")
def scatter_auto(inp, out):
    grid_x, grid_y = ttl.grid_size(dims=2)

    net = ttl.PipeNet(
        [ttl.Pipe(src=(x, 0), dst=(x, slice(1, grid_y))) for x in range(grid_x)]
    )

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as tile_in, out_cb.reserve() as tile_out:
            tile_out.store(ttl.math.abs(tile_in))

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with inp_cb.reserve() as blk:

            def read_and_send(pipe):
                tx = ttl.copy(inp[y, x], blk)
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


def test_scatter_auto(device):
    """Scatter with grid='auto': row 0 multicasts down each column."""
    grid = device.compute_with_storage_grid_size()
    grid_x, grid_y = grid.x, grid.y

    inp_torch = torch.randn(grid_y * TILE, grid_x * TILE, dtype=torch.bfloat16)

    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(
        torch.zeros(grid_y * TILE, grid_x * TILE, dtype=torch.bfloat16),
        device,
    )

    scatter_auto(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    # Each column's row-0 tile is broadcast to all rows, then abs applied.
    # So out[y, x] = abs(inp[0, x]) for all y.
    expected = torch.zeros_like(inp_torch)
    for x in range(grid_x):
        col_start = x * TILE
        col_end = (x + 1) * TILE
        tile0 = torch.abs(inp_torch[0:TILE, col_start:col_end])
        for y in range(grid_y):
            row_start = y * TILE
            row_end = (y + 1) * TILE
            expected[row_start:row_end, col_start:col_end] = tile0
    assert_pcc(expected, result)
