# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for uneven grid support - tensor dimensions that don't evenly divide by grid dimensions.

Verifies that TTL kernels work correctly when:
- Tensor tile count doesn't divide evenly by grid rows/cols
- Example: 64x64 tiles on a 2x3 grid (64 % 3 != 0)

This tests the 2D tensor shape approach which stores actual tile counts [tiles_y, tiles_x]
instead of 4D [grid_y, grid_x, shard_y, shard_x] which had ceiling division issues.
"""

import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

TILE_SIZE = 32

# Test various tensor shapes with grids that DON'T evenly divide them
# Format: (tensor_tiles_y, tensor_tiles_x, grid_cols, grid_rows)
# grid is (cols, rows) in tt-lang API
UNEVEN_GRID_CONFIGS = [
    # 64x64 tiles with grids that don't divide evenly
    (64, 64, 2, 3),  # 64 % 3 != 0
    (64, 64, 3, 2),  # 64 % 3 != 0
    (64, 64, 5, 7),  # 64 % 5 != 0, 64 % 7 != 0
    (64, 64, 7, 5),  # 64 % 7 != 0, 64 % 5 != 0
    # Smaller tensors with uneven grids
    (8, 8, 3, 3),  # 8 % 3 != 0
    (10, 10, 3, 4),  # 10 % 3 != 0, 10 % 4 != 0
    (16, 16, 5, 5),  # 16 % 5 != 0
    # Rectangular tensors
    (32, 64, 3, 5),  # 32 % 5 != 0, 64 % 3 != 0
    (64, 32, 5, 3),  # 64 % 5 != 0, 32 % 3 != 0
]


def tiles_to_shape(tiles_y: int, tiles_x: int) -> tuple[int, int]:
    """Convert tile counts to tensor shape in scalars."""
    return (tiles_y * TILE_SIZE, tiles_x * TILE_SIZE)


# Kernel template for uneven grid test
# Uses core coordinates to compute local work bounds with ceiling division
UNEVEN_GRID_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=({grid_cols}, {grid_rows}))
def uneven_grid_kernel(lhs, rhs, out):
    """Kernel that handles uneven grid distribution."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    # Total tiles in tensor
    total_tiles_y = {tiles_y}
    total_tiles_x = {tiles_x}

    # Grid dimensions
    grid_cols, grid_rows = ttl.grid_size(dims=2)

    # Tiles per core (ceiling division to handle uneven grids)
    tiles_per_core_y = -(-total_tiles_y // grid_rows)  # ceiling division
    tiles_per_core_x = -(-total_tiles_x // grid_cols)

    @ttl.compute()
    def compute():
        core_x, core_y = ttl.core(dims=2)
        start_y = core_y * tiles_per_core_y
        start_x = core_x * tiles_per_core_x

        for local_y in range(tiles_per_core_y):
            tile_y = start_y + local_y
            if tile_y < total_tiles_y:
                for local_x in range(tiles_per_core_x):
                    tile_x = start_x + local_x
                    if tile_x < total_tiles_x:
                        with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
                            o.store(l + r)

    @ttl.datamovement()
    def dm_read():
        core_x, core_y = ttl.core(dims=2)
        start_y = core_y * tiles_per_core_y
        start_x = core_x * tiles_per_core_x

        for local_y in range(tiles_per_core_y):
            tile_y = start_y + local_y
            if tile_y < total_tiles_y:
                for local_x in range(tiles_per_core_x):
                    tile_x = start_x + local_x
                    if tile_x < total_tiles_x:
                        with lhs_cb.reserve() as lhs_blk, rhs_cb.reserve() as rhs_blk:
                            tx_lhs = ttl.copy(lhs[tile_y, tile_x], lhs_blk)
                            tx_rhs = ttl.copy(rhs[tile_y, tile_x], rhs_blk)
                            tx_lhs.wait()
                            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, core_y = ttl.core(dims=2)
        start_y = core_y * tiles_per_core_y
        start_x = core_x * tiles_per_core_x

        for local_y in range(tiles_per_core_y):
            tile_y = start_y + local_y
            if tile_y < total_tiles_y:
                for local_x in range(tiles_per_core_x):
                    tile_x = start_x + local_x
                    if tile_x < total_tiles_x:
                        with out_cb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[tile_y, tile_x])
                            tx.wait()

'''

_kernel_cache = {}


def make_kernel(tiles_y: int, tiles_x: int, grid_cols: int, grid_rows: int):
    """Generate a kernel for the given configuration."""
    cache_key = (tiles_y, tiles_x, grid_cols, grid_rows)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    code = UNEVEN_GRID_KERNEL_TEMPLATE.format(
        tiles_y=tiles_y,
        tiles_x=tiles_x,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{tiles_y}x{tiles_x}_{grid_cols}x{grid_rows}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.uneven_grid_kernel
    _kernel_cache[cache_key] = kernel
    return kernel


@pytest.mark.parametrize(
    "config",
    UNEVEN_GRID_CONFIGS,
    ids=[f"{ty}x{tx}_grid{gc}x{gr}" for ty, tx, gc, gr in UNEVEN_GRID_CONFIGS],
)
def test_uneven_grid(device, config):
    """Test kernel execution with uneven grid distribution."""
    tiles_y, tiles_x, grid_cols, grid_rows = config
    height, width = tiles_to_shape(tiles_y, tiles_x)
    kernel = make_kernel(tiles_y, tiles_x, grid_cols, grid_rows)

    # Simple add: lhs + rhs
    lhs_torch = torch.full((height, width), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((height, width), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch  # Should be all 5.0

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert torch.allclose(
        result, expected, rtol=1e-2, atol=1e-2
    ), f"Uneven grid test failed for {tiles_y}x{tiles_x} tiles on {grid_cols}x{grid_rows} grid"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
