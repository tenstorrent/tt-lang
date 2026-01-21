# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for multicore kernel execution with core(dims=2) indexing.

Each core processes 2x2 tiles using dynamic indices based on its grid position.
Parameterized over grid shapes up to 8x8 cores (hardware limit).
"""

import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram

TILE_SIZE = 32
TILES_PER_CORE_ROW = 2  # Each core processes 2x2 tiles
TILES_PER_CORE_COL = 2

# Grid shapes limited to 8x8 (hardware limit)
GRID_SHAPES = [(r, c) for r in range(1, 8) for c in range(1, 8)]


def grid_to_tensor_shape(grid_rows: int, grid_cols: int) -> tuple[int, int]:
    """Convert grid dimensions to tensor shape (2x2 tiles per core)."""
    return (
        grid_rows * TILES_PER_CORE_ROW * TILE_SIZE,
        grid_cols * TILES_PER_CORE_COL * TILE_SIZE,
    )


# Multicore kernel template: each core processes 2x2 tiles via nested loop
MULTICORE_LOOP_KERNEL_TEMPLATE = '''
import ttl

# Grid: {grid_cols} cols x {grid_rows} rows
@ttl.kernel(grid=({grid_cols}, {grid_rows}))  # (cols, rows)
def multicore_loop(lhs, rhs, out):
    """Multicore kernel: each core loops over 2x2 tiles computing exp(lhs) + sqrt(rhs)."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fused_compute():
        for _ in range({tiles_per_core_row}):
            for _ in range({tiles_per_core_col}):
                with lhs_cb.wait() as lhs_tile, rhs_cb.wait() as rhs_tile:
                    with out_cb.reserve() as out_tile:
                        result = ttl.math.exp(lhs_tile) + ttl.math.sqrt(rhs_tile)
                        out_tile.store(result)

    @ttl.datamovement()
    def dm_read():
        for local_r in range({tiles_per_core_row}):
            for local_c in range({tiles_per_core_col}):
                with lhs_cb.reserve() as lhs_blk, rhs_cb.reserve() as rhs_blk:
                    # core(dims=2) returns (x, y) where x=col, y=row
                    x, y = ttl.core(dims=2)
                    row = y * {tiles_per_core_row} + local_r
                    col = x * {tiles_per_core_col} + local_c
                    tx_lhs = ttl.copy(lhs[row, col], lhs_blk)
                    tx_rhs = ttl.copy(rhs[row, col], rhs_blk)
                    tx_lhs.wait()
                    tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        for local_r in range({tiles_per_core_row}):
            for local_c in range({tiles_per_core_col}):
                with out_cb.wait() as out_blk:
                    x, y = ttl.core(dims=2)
                    row = y * {tiles_per_core_row} + local_r
                    col = x * {tiles_per_core_col} + local_c
                    tx = ttl.copy(out_blk, out[row, col])
                    tx.wait()

'''

_kernel_cache = {}


def make_kernel(grid_rows: int, grid_cols: int):
    """Generate a multicore loop kernel for the given grid dimensions."""
    cache_key = (grid_rows, grid_cols)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    code = MULTICORE_LOOP_KERNEL_TEMPLATE.format(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        tiles_per_core_row=TILES_PER_CORE_ROW,
        tiles_per_core_col=TILES_PER_CORE_COL,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{grid_rows}x{grid_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.multicore_loop
    _kernel_cache[cache_key] = kernel
    return kernel


@pytest.mark.parametrize(
    "grid_shape",
    GRID_SHAPES,
    ids=[f"{r}x{c}" for r, c in GRID_SHAPES],
)
def test_multicore_loop(device, grid_shape):
    """Test multicore kernel with 2x2 tiles per core: exp(lhs) + sqrt(rhs)."""
    grid_rows, grid_cols = grid_shape
    height, width = grid_to_tensor_shape(grid_rows, grid_cols)
    kernel = make_kernel(grid_rows, grid_cols)

    # Random inputs: small values for lhs (exp overflow), positive for rhs (sqrt)
    lhs_torch = torch.rand((height, width), dtype=torch.bfloat16) * 0.5
    rhs_torch = torch.rand((height, width), dtype=torch.bfloat16) * 4.0 + 0.1
    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)
    expected = torch.exp(lhs_torch) + torch.sqrt(rhs_torch)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    # Verify grid_size returns correct values
    import ttl

    x_size, y_size = ttl.grid_size(dims=2)
    assert (x_size, y_size) == (grid_cols, grid_rows)

    assert torch.allclose(result.float(), expected.float(), rtol=0.02, atol=0.5)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
