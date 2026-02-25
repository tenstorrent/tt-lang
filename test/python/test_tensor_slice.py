# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for tensor slice indexing across multiple tensor shapes.

Tests that tensor[row, col] syntax correctly accesses specific tiles
within multi-tile tensors using loops over tile indices.

Parameterized over tensor shapes from 1x1 to 16x16 tiles.
"""

import importlib.util
import tempfile

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttlang_test_utils import to_dram

TILE_SIZE = 32

# Test shapes: all combinations from 1x1 to 16x16 tiles (512x512 elements max)
TENSOR_TILE_SHAPES = [(r, c) for r in range(1, 16) for c in range(1, 16)]
TENSOR_TILE_SHAPES_SHORT = [(r, c) for r in range(1, 5) for c in range(1, 5)]


def tiles_to_tensor_shape(tile_rows: int, tile_cols: int) -> tuple[int, int]:
    return (tile_rows * TILE_SIZE, tile_cols * TILE_SIZE)


ADD_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def tile_loop_kernel(inp, bias, out):
    """Kernel that loops over all tiles, adding bias to each."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        for _ in range({tile_rows}):
            for _ in range({tile_cols}):
                with inp_dfb.wait() as inp_tile, bias_dfb.wait() as bias_tile:
                    with out_dfb.reserve() as out_tile:
                        result = inp_tile + bias_tile
                        out_tile.store(result)

    @ttl.datamovement()
    def dm_read():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with inp_dfb.reserve() as inp_blk, bias_dfb.reserve() as bias_blk:
                    tx_inp = ttl.copy(inp[r, c], inp_blk)
                    tx_bias = ttl.copy(bias[r, c], bias_blk)
                    tx_inp.wait()
                    tx_bias.wait()

    @ttl.datamovement()
    def dm_write():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[r, c])
                    tx.wait()

'''

FUSED_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def fused_kernel(inp, bias, out):
    """Kernel that computes ttl.math.exp(inp) + ttl.math.sqrt(bias) - fuses 3 ops."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fused_compute():
        for _ in range({tile_rows}):
            for _ in range({tile_cols}):
                with inp_dfb.wait() as inp_tile, bias_dfb.wait() as bias_tile:
                    with out_dfb.reserve() as out_tile:
                        result = ttl.math.exp(inp_tile) + ttl.math.sqrt(bias_tile)
                        out_tile.store(result)

    @ttl.datamovement()
    def dm_read():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with inp_dfb.reserve() as inp_blk, bias_dfb.reserve() as bias_blk:
                    tx_inp = ttl.copy(inp[r, c], inp_blk)
                    tx_bias = ttl.copy(bias[r, c], bias_blk)
                    tx_inp.wait()
                    tx_bias.wait()

    @ttl.datamovement()
    def dm_write():
        for r in range({tile_rows}):
            for c in range({tile_cols}):
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[r, c])
                    tx.wait()

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
    temp_kernel_files.append(temp_path)

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
    temp_kernel_files.append(temp_path)

    kernel = module.fused_kernel
    _fused_kernel_cache[cache_key] = kernel
    return kernel


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

    bias_torch = torch.full((height, width), 100.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    bias = to_dram(bias_torch, device)
    out = to_dram(out_torch, device)

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
            )


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
            tile_value = float(r * tile_cols + c + 1) * 0.1
            inp_torch[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ] = tile_value

    bias_torch = torch.full((height, width), 4.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    bias = to_dram(bias_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, bias, out)
    result = ttnn.to_torch(out)

    # Verify each tile: exp(inp) + sqrt(bias) = exp(tile_value) + sqrt(4.0)
    sqrt_bias = 2.0
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
            )


# =============================================================================
# Block DFB tensor slice tests - DFB shape > (1, 1) to trigger DST subblocking
# =============================================================================

# Format: (tile_rows, tile_cols, block_r, block_c)
BLOCK_ADD_CONFIGS = [
    (8, 8, 4, 4),  # 4x4 block on 8x8 tensor (16 tiles per block)
    (12, 12, 4, 4),  # 4x4 block on 12x12 tensor (16 tiles per block)
    (12, 12, 3, 3),  # 3x3 block on 12x12 tensor (9 tiles per block)
    (16, 16, 4, 4),  # 4x4 block on 16x16 tensor (16 tiles per block)
]

BLOCK_ADD_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def block_add_kernel(inp, bias, out):
    """Kernel with block DFB that loops over blocks, adding bias."""
    BLOCK_R = {block_r}
    BLOCK_C = {block_c}

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(BLOCK_R, BLOCK_C), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(BLOCK_R, BLOCK_C), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(BLOCK_R, BLOCK_C), buffer_factor=2)

    blocks_r = {tile_rows} // BLOCK_R
    blocks_c = {tile_cols} // BLOCK_C

    @ttl.compute()
    def add_compute():
        for _ in range(blocks_r):
            for _ in range(blocks_c):
                with inp_dfb.wait() as inp_tile, bias_dfb.wait() as bias_tile:
                    with out_dfb.reserve() as out_tile:
                        result = inp_tile + bias_tile
                        out_tile.store(result)

    @ttl.datamovement()
    def dm_read():
        for br in range(blocks_r):
            for bc in range(blocks_c):
                start_r = br * BLOCK_R
                start_c = bc * BLOCK_C
                with inp_dfb.reserve() as inp_blk, bias_dfb.reserve() as bias_blk:
                    tx_inp = ttl.copy(inp[start_r:start_r + BLOCK_R, start_c:start_c + BLOCK_C], inp_blk)
                    tx_bias = ttl.copy(bias[start_r:start_r + BLOCK_R, start_c:start_c + BLOCK_C], bias_blk)
                    tx_inp.wait()
                    tx_bias.wait()

    @ttl.datamovement()
    def dm_write():
        for br in range(blocks_r):
            for bc in range(blocks_c):
                start_r = br * BLOCK_R
                start_c = bc * BLOCK_C
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[start_r:start_r + BLOCK_R, start_c:start_c + BLOCK_C])
                    tx.wait()

'''

_block_add_kernel_cache = {}


def make_block_add_kernel(tile_rows: int, tile_cols: int, block_r: int, block_c: int):
    """Generate a block DFB add kernel."""
    cache_key = (tile_rows, tile_cols, block_r, block_c)
    if cache_key in _block_add_kernel_cache:
        return _block_add_kernel_cache[cache_key]

    code = BLOCK_ADD_KERNEL_TEMPLATE.format(
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        block_r=block_r,
        block_c=block_c,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"block_add_{tile_rows}x{tile_cols}_blk{block_r}x{block_c}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("block_add_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    temp_kernel_files.append(temp_path)

    kernel = module.block_add_kernel
    _block_add_kernel_cache[cache_key] = kernel
    return kernel


@pytest.mark.parametrize(
    "config",
    BLOCK_ADD_CONFIGS,
    ids=[f"{tr}x{tc}_blk{br}x{bc}" for tr, tc, br, bc in BLOCK_ADD_CONFIGS],
)
def test_tensor_slice_block_add(device, config, compiler_options):
    """Test block DFB add kernel (>8 tiles per block) to exercise DST subblocking."""
    tile_rows, tile_cols, block_r, block_c = config
    height, width = tiles_to_tensor_shape(tile_rows, tile_cols)
    kernel = make_block_add_kernel(tile_rows, tile_cols, block_r, block_c)

    # Input tensor with unique value per tile for verification
    inp_torch = torch.zeros((height, width), dtype=torch.bfloat16)
    for r in range(tile_rows):
        for c in range(tile_cols):
            tile_value = float(r * tile_cols + c + 1)
            inp_torch[
                r * TILE_SIZE : (r + 1) * TILE_SIZE,
                c * TILE_SIZE : (c + 1) * TILE_SIZE,
            ] = tile_value

    bias_torch = torch.full((height, width), 100.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((height, width), dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    bias = to_dram(bias_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, bias, out, options=compiler_options)
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
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
