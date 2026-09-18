# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test the flattened (dims=1) forms of ttl.node and ttl.grid_size.

The specification flattens the highest-rank dimension into the one below it, so
for a (cols, rows) grid the row coordinate varies fastest:
ttl.node(dims=1) == x * rows + y and ttl.grid_size(dims=1) == cols * rows.

The grid is deliberately non-square so that transposing the flattening order
produces a detectably permuted result rather than an identical one.
"""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import torch

import ttl
from ttlang_test_utils import to_dram

TILE_SIZE = 32
GRID_COLS = 4
GRID_ROWS = 2


@ttl.operation(grid=(GRID_COLS, GRID_ROWS))
def flattened_index_copy(a, out):
    """Copy a[y, x] to out[y, x], addressing the source through node(dims=1)."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a_tile, out_dfb.reserve() as out_tile:
            out_tile.store(a_tile)

    @ttl.datamovement()
    def dm_read():
        node = ttl.node(dims=1)
        with a_dfb.reserve() as a_blk:
            tx = ttl.copy(a[node % GRID_ROWS, node // GRID_ROWS], a_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


def test_node_dims1_flattens_with_row_fastest(device):
    """A node reached through its flattened index sees its own (x, y) tile."""
    shape = (GRID_ROWS * TILE_SIZE, GRID_COLS * TILE_SIZE)

    # Give every tile a distinct value so a transposed flattening permutes the
    # output instead of reproducing it.
    source = torch.zeros(shape, dtype=torch.bfloat16)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            tile = row * GRID_COLS + col
            source[
                row * TILE_SIZE : (row + 1) * TILE_SIZE,
                col * TILE_SIZE : (col + 1) * TILE_SIZE,
            ] = (
                tile + 1
            )

    a = to_dram(source, device)
    out = to_dram(torch.zeros(shape, dtype=torch.bfloat16), device)

    flattened_index_copy(a, out)

    result = ttnn.to_torch(out).to(torch.bfloat16)
    assert torch.equal(result, source)


def test_grid_size_dims1_is_the_node_count(device):
    """The flattened grid size is the product of the grid extents."""
    shape = (GRID_ROWS * TILE_SIZE, GRID_COLS * TILE_SIZE)
    a = to_dram(torch.ones(shape, dtype=torch.bfloat16), device)
    out = to_dram(torch.zeros(shape, dtype=torch.bfloat16), device)

    flattened_index_copy(a, out)

    assert ttl.grid_size(dims=1) == GRID_COLS * GRID_ROWS


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
