# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for multicore kernel execution with core_x() and core_y() indexing.

Each core processes 2x2 tiles using dynamic indices based on its grid position.
Parameterized over grid shapes up to 8x8 cores (hardware limit).
"""

import pytest
import torch
import tempfile
import importlib.util

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")

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
from ttlang import ttl

@ttl.kernel(grid=({grid_rows}, {grid_cols}))
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
                    # Compute global indices: core * tiles_per_core + local
                    row = ttl.core_y() * {tiles_per_core_row} + local_r
                    col = ttl.core_x() * {tiles_per_core_col} + local_c
                    tx_lhs = ttl.copy(lhs[row, col], lhs_blk)
                    tx_rhs = ttl.copy(rhs[row, col], rhs_blk)
                    tx_lhs.wait()
                    tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        for local_r in range({tiles_per_core_row}):
            for local_c in range({tiles_per_core_col}):
                with out_cb.wait() as out_blk:
                    row = ttl.core_y() * {tiles_per_core_row} + local_r
                    col = ttl.core_x() * {tiles_per_core_col} + local_c
                    tx = ttl.copy(out_blk, out[row, col])
                    tx.wait()

    return ttl.Program(fused_compute, dm_read, dm_write)(lhs, rhs, out)
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


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize(
    "grid_shape",
    GRID_SHAPES,
    ids=[f"{r}x{c}" for r, c in GRID_SHAPES],
)
def test_multicore_loop(device, grid_shape):
    """Test multicore kernel with 2x2 tiles per core: exp(lhs) + sqrt(rhs)."""
    grid_rows, grid_cols = grid_shape
    height, width = grid_to_tensor_shape(grid_rows, grid_cols)
    total_tile_rows = grid_rows * TILES_PER_CORE_ROW
    total_tile_cols = grid_cols * TILES_PER_CORE_COL

    kernel = make_kernel(grid_rows, grid_cols)

    # Input tensors with small values for exp (avoid overflow)
    lhs_torch = torch.zeros((height, width), dtype=torch.bfloat16)
    rhs_torch = torch.zeros((height, width), dtype=torch.bfloat16)

    for r in range(total_tile_rows):
        for c in range(total_tile_cols):
            # Use 0.01 multiplier to keep exp() output small for bfloat16 precision
            lhs_value = float(r * total_tile_cols + c + 1) * 0.01
            rhs_value = 4.0
            lhs_torch[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ] = lhs_value
            rhs_torch[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ] = rhs_value

    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)

    lhs = ttnn.from_torch(
        lhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rhs = ttnn.from_torch(
        rhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    kernel(lhs, rhs, out)

    result = ttnn.to_torch(out)

    # Verify each tile: exp(lhs) + sqrt(rhs)
    sqrt_rhs = 2.0
    for r in range(total_tile_rows):
        for c in range(total_tile_cols):
            tile_value = float(r * total_tile_cols + c + 1) * 0.01
            expected_value = float(torch.exp(torch.tensor(tile_value))) + sqrt_rhs
            result_tile = result[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ]

            assert torch.allclose(
                result_tile.float(),
                torch.full((TILE_SIZE, TILE_SIZE), expected_value),
                rtol=0.02,
                atol=0.5,
            ), f"Tile [{r}, {c}]: expected {expected_value}, got {result_tile[0,0].item()}"


if __name__ == "__main__":
    import sys

    if not TTNN_AVAILABLE:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
