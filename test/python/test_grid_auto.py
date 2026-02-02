# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test that grid='auto' resolves to device compute grid dimensions.
"""

import pytest

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import torch

import ttl
from ttlang_test_utils import to_dram


TILE_SIZE = 32


@ttl.kernel(grid="auto")
def auto_grid_kernel(a, out):
    """Simple kernel using automatic grid sizing."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, out_cb.reserve() as out_tile:
            out_tile.store(a_tile + a_tile)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        with a_cb.reserve() as a_blk:
            tx = ttl.copy(a[y, x], a_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


def test_grid_auto(device):
    """Verify grid='auto' resolves to full device compute grid."""
    device_grid = device.compute_with_storage_grid_size()
    expected_cols, expected_rows = device_grid.x, device_grid.y

    # Shape: one tile per core
    shape = (expected_rows * TILE_SIZE, expected_cols * TILE_SIZE)

    a = to_dram(torch.ones(shape, dtype=torch.bfloat16), device)
    out = to_dram(torch.zeros(shape, dtype=torch.bfloat16), device)

    auto_grid_kernel(a, out)

    # Verify grid_size returns device compute grid
    actual_cols, actual_rows = ttl.grid_size(dims=2)
    assert (actual_cols, actual_rows) == (expected_cols, expected_rows)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
