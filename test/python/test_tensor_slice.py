# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for tensor slice indexing across multiple tensor shapes.

Tests that tensor[row, col] syntax correctly accesses specific tiles
within multi-tile tensors using loops over tile indices.

Parameterized over tensor shapes from 1x1 to 16x16 tiles.
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

# Test shapes: all combinations from 1x1 to 16x16 tiles (512x512 elements max)
TENSOR_TILE_SHAPES = [(r, c) for r in range(1, 16) for c in range(1, 16)]
TENSOR_TILE_SHAPES_SHORT = [(r, c) for r in range(1, 5) for c in range(1, 5)]


def tiles_to_tensor_shape(tile_rows: int, tile_cols: int) -> tuple[int, int]:
    return (tile_rows * TILE_SIZE, tile_cols * TILE_SIZE)


ADD_KERNEL_TEMPLATE = '''
from ttlang import ttl, make_circular_buffer_like
from ttlang.ttl_api import Program
from ttlang.operators import copy

@ttl.kernel(grid=(1, 1))
def tile_loop_kernel(inp, bias, out):
    """Kernel that loops over all tiles, adding bias to each."""
    inp_cb = make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    bias_cb = make_circular_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        for _ in range({tile_rows}):
            for _ in range({tile_cols}):
                with inp_cb.wait() as inp_tile, bias_cb.wait() as bias_tile:
                    with out_cb.reserve() as out_tile:
                        result = inp_tile + bias_tile
                        out_tile.store(result)

    @ttl.datamovement()
    def dm_read():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with inp_cb.reserve(), bias_cb.reserve():
                    tx_inp = copy(inp[r, c], inp_cb)
                    tx_bias = copy(bias[r, c], bias_cb)
                    tx_inp.wait()
                    tx_bias.wait()

    @ttl.datamovement()
    def dm_write():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with out_cb.wait():
                    tx = copy(out_cb, out[r, c])
                    tx.wait()

    return Program(add_compute, dm_read, dm_write)(inp, bias, out)
'''

FUSED_KERNEL_TEMPLATE = '''
from ttlang import ttl, make_circular_buffer_like, exp, sqrt
from ttlang.ttl_api import Program
from ttlang.operators import copy

@ttl.kernel(grid=(1, 1))
def fused_kernel(inp, bias, out):
    """Kernel that computes exp(inp) + sqrt(bias) - fuses 3 ops."""
    inp_cb = make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    bias_cb = make_circular_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fused_compute():
        for _ in range({tile_rows}):
            for _ in range({tile_cols}):
                with inp_cb.wait() as inp_tile, bias_cb.wait() as bias_tile:
                    with out_cb.reserve() as out_tile:
                        result = exp(inp_tile) + sqrt(bias_tile)
                        out_tile.store(result)

    @ttl.datamovement()
    def dm_read():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with inp_cb.reserve(), bias_cb.reserve():
                    tx_inp = copy(inp[r, c], inp_cb)
                    tx_bias = copy(bias[r, c], bias_cb)
                    tx_inp.wait()
                    tx_bias.wait()

    @ttl.datamovement()
    def dm_write():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with out_cb.wait():
                    tx = copy(out_cb, out[r, c])
                    tx.wait()

    return Program(fused_compute, dm_read, dm_write)(inp, bias, out)
'''

_add_kernel_cache = {}
_fused_kernel_cache = {}


def make_add_kernel(tile_rows: int, tile_cols: int):
    """Generate an add kernel for the given tile dimensions."""
    cache_key = (tile_rows, tile_cols)
    if cache_key in _add_kernel_cache:
        return _add_kernel_cache[cache_key]

    code = ADD_KERNEL_TEMPLATE.format(tile_rows=tile_rows, tile_cols=tile_cols)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_add_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("add_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.tile_loop_kernel
    _add_kernel_cache[cache_key] = kernel
    return kernel


def make_fused_kernel(tile_rows: int, tile_cols: int):
    """Generate a fused exp+sqrt+add kernel for the given tile dimensions."""
    cache_key = (tile_rows, tile_cols)
    if cache_key in _fused_kernel_cache:
        return _fused_kernel_cache[cache_key]

    code = FUSED_KERNEL_TEMPLATE.format(tile_rows=tile_rows, tile_cols=tile_cols)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_fused_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("fused_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.fused_kernel
    _fused_kernel_cache[cache_key] = kernel
    return kernel


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def make_test_id(shape):
    return f"{shape[0]}x{shape[1]}tiles"


@pytest.mark.parametrize(
    "tensor_shape",
    TENSOR_TILE_SHAPES,
    ids=[make_test_id(s) for s in TENSOR_TILE_SHAPES],
)
def test_tensor_slice_add(device, tensor_shape):
    """Test looping over all tiles with add operation."""
    tile_rows, tile_cols = tensor_shape
    height, width = tiles_to_tensor_shape(tile_rows, tile_cols)

    kernel = make_add_kernel(tile_rows, tile_cols)

    # Input tensor with unique value per tile for verification
    inp_torch = torch.zeros((height, width), dtype=torch.bfloat16)
    for r in range(tile_rows):
        for c in range(tile_cols):
            tile_value = float(r * tile_cols + c + 1)
            inp_torch[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ] = tile_value

    # Bias tensor - constant value
    bias_torch = torch.full((height, width), 100.0, dtype=torch.bfloat16)

    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)

    inp = ttnn.from_torch(
        inp_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        bias_torch,
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

    kernel(inp, bias, out)

    result = ttnn.to_torch(out)

    # Verify each tile was correctly processed (inp + bias)
    for r in range(tile_rows):
        for c in range(tile_cols):
            expected_value = float(r * tile_cols + c + 1) + 100.0
            result_tile = result[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ]
            expected_tile = torch.full((TILE_SIZE, TILE_SIZE), expected_value)

            assert torch.allclose(
                result_tile.float(), expected_tile.float(), rtol=1e-2, atol=1e-2
            ), f"Tile at [{r}, {c}] mismatch: expected {expected_value}, got {result_tile[0,0].item()}"


@pytest.mark.parametrize(
    "tensor_shape",
    TENSOR_TILE_SHAPES_SHORT,
    ids=[make_test_id(s) for s in TENSOR_TILE_SHAPES_SHORT],
)
def test_tensor_slice_fused(device, tensor_shape):
    """Test looping over all tiles with fused exp(inp) + sqrt(bias) operation."""
    tile_rows, tile_cols = tensor_shape
    height, width = tiles_to_tensor_shape(tile_rows, tile_cols)

    kernel = make_fused_kernel(tile_rows, tile_cols)

    # Input tensor with small values to avoid exp overflow
    inp_torch = torch.zeros((height, width), dtype=torch.bfloat16)
    for r in range(tile_rows):
        for c in range(tile_cols):
            # Use small values (0.1, 0.2, ...) to keep exp() reasonable
            tile_value = float(r * tile_cols + c + 1) * 0.1
            inp_torch[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ] = tile_value

    # Bias tensor - constant positive value for sqrt
    bias_torch = torch.full((height, width), 4.0, dtype=torch.bfloat16)

    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)

    inp = ttnn.from_torch(
        inp_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bias = ttnn.from_torch(
        bias_torch,
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

    kernel(inp, bias, out)

    result = ttnn.to_torch(out)

    # Verify each tile: exp(inp) + sqrt(bias) = exp(tile_value) + sqrt(4.0)
    sqrt_bias = 2.0  # sqrt(4.0)
    for r in range(tile_rows):
        for c in range(tile_cols):
            tile_value = float(r * tile_cols + c + 1) * 0.1
            expected_value = float(torch.exp(torch.tensor(tile_value))) + sqrt_bias
            result_tile = result[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ]
            expected_tile = torch.full((TILE_SIZE, TILE_SIZE), expected_value)

            assert torch.allclose(
                result_tile.float(), expected_tile.float(), rtol=1e-2, atol=1e-2
            ), f"Tile at [{r}, {c}] mismatch: expected {expected_value}, got {result_tile[0,0].item()}"


if __name__ == "__main__":
    import sys

    if not TTNN_AVAILABLE:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    print("=== Tensor Slice Loop Test ===")
    print(f"Tensor shapes: {TENSOR_TILE_SHAPES}")

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
