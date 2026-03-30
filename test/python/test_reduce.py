# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTL reduce operations (reduce_sum, reduce_max).

Uses dynamically generated kernels (string templates + importlib) to
parameterize over block shapes, reduction dimensions, and reduce functions.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import atexit
import importlib
import os
import tempfile
from typing import Callable, List, Tuple

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

TILE = 32

# =============================================================================
# Kernel generation from templates
# =============================================================================

REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def reduce_kernel(inp, scaler, out):
    """Reduce {reduce_fn} dims={dims} on ({inp_rows},{inp_cols}) grid."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({inp_rows}, {inp_cols}), buffer_factor=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({out_rows}, {out_cols}), buffer_factor=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, scaler_dfb.wait() as scaler_blk, out_dfb.reserve() as out_blk:
            result = ttl.math.{reduce_fn}(inp_blk, scaler_blk, dims={dims})
            out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[{inp_slice}], inp_blk)
        tx_inp.wait()
        inp_blk.push()
        scaler_blk = scaler_dfb.reserve()
        tx_scaler = ttl.copy(scaler[0, 0], scaler_blk)
        tx_scaler.wait()
        scaler_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[{out_slice}])
        tx_out.wait()
        out_blk.pop()
'''

MULTICORE_REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=({grid_cols}, {grid_rows}))
def reduce_kernel(inp, scaler, out):
    """Multicore reduce {reduce_fn} dims={dims}, each core reduces its own tile."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=(1, 1), buffer_factor=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), buffer_factor=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, scaler_dfb.wait() as scaler_blk, out_dfb.reserve() as out_blk:
            result = ttl.math.{reduce_fn}(inp_blk, scaler_blk, dims={dims})
            out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        core_x, core_y = ttl.node(dims=2)
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[core_y, core_x], inp_blk)
        tx_inp.wait()
        inp_blk.push()
        scaler_blk = scaler_dfb.reserve()
        tx_scaler = ttl.copy(scaler[0, 0], scaler_blk)
        tx_scaler.wait()
        scaler_blk.push()

    @ttl.datamovement()
    def dm_write():
        core_x, core_y = ttl.node(dims=2)
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[core_y, core_x])
        tx_out.wait()
        out_blk.pop()
'''

_kernel_cache = {}
_temp_files = []


def _slice_syntax(rows: int, cols: int) -> str:
    """Generate tensor slice syntax for a tile grid."""
    if rows == 1 and cols == 1:
        return "0, 0"
    return f"0:{rows}, 0:{cols}"


def _compute_out_shape(
    inp_rows: int, inp_cols: int, dims: List[int]
) -> Tuple[int, int]:
    """Compute output tile grid shape after reduction."""
    norm = {dim % 2 for dim in dims}
    out_rows = 1 if 0 in norm else inp_rows
    out_cols = 1 if 1 in norm else inp_cols
    return out_rows, out_cols


def make_reduce_kernel(
    reduce_fn: str, inp_rows: int, inp_cols: int, dims: List[int]
) -> Callable:
    """Generate and cache a reduce kernel for the given configuration."""
    out_rows, out_cols = _compute_out_shape(inp_rows, inp_cols, dims)
    cache_key = (reduce_fn, inp_rows, inp_cols, tuple(dims))
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    code = REDUCE_KERNEL_TEMPLATE.format(
        reduce_fn=reduce_fn,
        inp_rows=inp_rows,
        inp_cols=inp_cols,
        out_rows=out_rows,
        out_cols=out_cols,
        dims=dims,
        inp_slice=_slice_syntax(inp_rows, inp_cols),
        out_slice=_slice_syntax(out_rows, out_cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"reduce_{reduce_fn}_{inp_rows}x{inp_cols}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("reduce_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.reduce_kernel
    _kernel_cache[cache_key] = kernel
    return kernel


def make_multicore_reduce_kernel(
    reduce_fn: str, grid_rows: int, grid_cols: int, dims: List[int]
) -> Callable:
    """Generate a multicore reduce kernel (1 tile per core)."""
    cache_key = ("multicore", reduce_fn, grid_rows, grid_cols, tuple(dims))
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    code = MULTICORE_REDUCE_KERNEL_TEMPLATE.format(
        reduce_fn=reduce_fn,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        dims=dims,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"mc_reduce_{grid_rows}x{grid_cols}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("mc_reduce_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.reduce_kernel
    _kernel_cache[cache_key] = kernel
    return kernel


def _cleanup_temp_files():
    for path in _temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_files)


# =============================================================================
# Scaler helper
# =============================================================================


def create_scaler_tile(value: float = 1.0, dtype=torch.bfloat16):
    """Scaler tile with value in first row of each 16x16 face."""
    tile = torch.zeros((TILE, TILE), dtype=dtype)
    tile[0, :] = value
    tile[16, :] = value
    return tile


# =============================================================================
# Test configurations
# =============================================================================

# (reduce_fn, inp_shape, dims, inp_factory, scaler_val, description)
SINGLE_TILE_CONFIGS = [
    # reduce_sum dim 0 (reduce rows / height -> REDUCE_COL)
    (
        "reduce_sum",
        (1, 1),
        [0],
        lambda: torch.ones(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_dim0_ones",
    ),
    (
        "reduce_sum",
        (1, 1),
        [0],
        lambda: torch.arange(TILE, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(TILE, TILE)
        .contiguous(),
        1.0,
        "sum_dim0_ascending",
    ),
    (
        "reduce_sum",
        (1, 1),
        [0],
        lambda: torch.ones(TILE, TILE, dtype=torch.bfloat16),
        0.5,
        "sum_dim0_scaler_half",
    ),
    # reduce_sum dim 1 (reduce cols / width -> REDUCE_ROW)
    (
        "reduce_sum",
        (1, 1),
        [1],
        lambda: torch.ones(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_dim1_ones",
    ),
    (
        "reduce_sum",
        (1, 1),
        [1],
        lambda: torch.arange(TILE, dtype=torch.bfloat16)
        .unsqueeze(0)
        .expand(TILE, TILE)
        .contiguous(),
        1.0,
        "sum_dim1_ascending",
    ),
    (
        "reduce_sum",
        (1, 1),
        [1],
        lambda: torch.ones(TILE, TILE, dtype=torch.bfloat16),
        0.5,
        "sum_dim1_scaler_half",
    ),
    # reduce_sum both dims
    (
        "reduce_sum",
        (1, 1),
        [0, 1],
        lambda: torch.ones(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_both_ones",
    ),
    # reduce_sum negative dims
    (
        "reduce_sum",
        (1, 1),
        [-1],
        lambda: torch.ones(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_neg1_ones",
    ),
    (
        "reduce_sum",
        (1, 1),
        [-2],
        lambda: torch.ones(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_neg2_ones",
    ),
    # reduce_max dim 0
    (
        "reduce_max",
        (1, 1),
        [0],
        lambda: torch.arange(TILE, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(TILE, TILE)
        .contiguous(),
        1.0,
        "max_dim0_ascending",
    ),
    # reduce_max dim 1
    (
        "reduce_max",
        (1, 1),
        [1],
        lambda: torch.arange(TILE, dtype=torch.bfloat16)
        .unsqueeze(0)
        .expand(TILE, TILE)
        .contiguous(),
        1.0,
        "max_dim1_ascending",
    ),
    (
        "reduce_max",
        (1, 1),
        [1],
        lambda: torch.zeros(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "max_dim1_zeros",
    ),
    (
        "reduce_max",
        (1, 1),
        [1],
        lambda: (torch.arange(TILE, dtype=torch.bfloat16) - 16)
        .unsqueeze(0)
        .expand(TILE, TILE)
        .contiguous(),
        1.0,
        "max_dim1_negatives",
    ),
    # Random inputs for each dimension combination.
    (
        "reduce_sum",
        (1, 1),
        [0],
        lambda: torch.rand(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_dim0_random",
    ),
    (
        "reduce_sum",
        (1, 1),
        [1],
        lambda: torch.rand(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_dim1_random",
    ),
    (
        "reduce_sum",
        (1, 1),
        [0, 1],
        lambda: torch.rand(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "sum_both_random",
    ),
    (
        "reduce_max",
        (1, 1),
        [0],
        lambda: torch.rand(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "max_dim0_random",
    ),
    (
        "reduce_max",
        (1, 1),
        [1],
        lambda: torch.rand(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "max_dim1_random",
    ),
    (
        "reduce_max",
        (1, 1),
        [0, 1],
        lambda: torch.rand(TILE, TILE, dtype=torch.bfloat16),
        1.0,
        "max_both_random",
    ),
]

MULTI_TILE_CONFIGS = [
    (
        "reduce_sum",
        (2, 2),
        [0],
        lambda: torch.ones(64, 64, dtype=torch.bfloat16),
        1.0,
        "sum_2x2_dim0",
    ),
    (
        "reduce_sum",
        (2, 2),
        [1],
        lambda: torch.ones(64, 64, dtype=torch.bfloat16),
        1.0,
        "sum_2x2_dim1",
    ),
    (
        "reduce_sum",
        (2, 2),
        [0, 1],
        lambda: torch.ones(64, 64, dtype=torch.bfloat16),
        1.0,
        "sum_2x2_both",
    ),
    # Random multi-tile.
    (
        "reduce_sum",
        (2, 2),
        [0],
        lambda: torch.rand(64, 64, dtype=torch.bfloat16),
        1.0,
        "sum_2x2_dim0_random",
    ),
    (
        "reduce_max",
        (2, 2),
        [1],
        lambda: torch.rand(64, 64, dtype=torch.bfloat16),
        1.0,
        "max_2x2_dim1_random",
    ),
    # Large block (4x4).
    (
        "reduce_sum",
        (4, 4),
        [0],
        lambda: torch.ones(128, 128, dtype=torch.bfloat16),
        1.0,
        "sum_4x4_dim0_ones",
    ),
    (
        "reduce_sum",
        (4, 4),
        [0],
        lambda: torch.rand(128, 128, dtype=torch.bfloat16),
        1.0,
        "sum_4x4_dim0_random",
    ),
    (
        "reduce_max",
        (4, 4),
        [1],
        lambda: torch.rand(128, 128, dtype=torch.bfloat16),
        1.0,
        "max_4x4_dim1_random",
    ),
    (
        "reduce_sum",
        (4, 4),
        [0, 1],
        lambda: torch.rand(128, 128, dtype=torch.bfloat16),
        1.0,
        "sum_4x4_both_random",
    ),
]


def _expected_reduce_value(inp_torch, reduce_fn, dims, scaler_val):
    """Compute expected value at position [0, 0] of the reduced output."""
    norm_dims = sorted({dim % 2 for dim in dims})
    val = inp_torch.float()
    if reduce_fn == "reduce_sum":
        for dim in norm_dims:
            val = val.sum(dim=dim, keepdim=True)
    else:
        for dim in norm_dims:
            val = val.amax(dim=dim, keepdim=True)
    return (val * scaler_val).flatten()[0].item()


# =============================================================================
# Parameterized tests
# =============================================================================


@pytest.mark.parametrize(
    "reduce_fn, inp_shape, dims, inp_factory, scaler_val, test_id",
    SINGLE_TILE_CONFIGS,
    ids=[cfg[-1] for cfg in SINGLE_TILE_CONFIGS],
)
def test_reduce_single_tile(
    device, reduce_fn, inp_shape, dims, inp_factory, scaler_val, test_id
):
    """Single-tile reduce with parameterized inputs."""
    inp_rows, inp_cols = inp_shape
    kernel = make_reduce_kernel(reduce_fn, inp_rows, inp_cols, dims)

    inp_torch = inp_factory()
    scaler_torch = create_scaler_tile(scaler_val)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    expected_val = _expected_reduce_value(inp_torch, reduce_fn, dims, scaler_val)
    assert result[0, 0].float().item() == pytest.approx(expected_val, rel=0.05, abs=1.0)


@pytest.mark.parametrize(
    "reduce_fn, inp_shape, dims, inp_factory, scaler_val, test_id",
    MULTI_TILE_CONFIGS,
    ids=[cfg[-1] for cfg in MULTI_TILE_CONFIGS],
)
def test_reduce_multi_tile(
    device, reduce_fn, inp_shape, dims, inp_factory, scaler_val, test_id
):
    """Multi-tile reduce with parameterized grid shapes."""
    inp_rows, inp_cols = inp_shape
    out_rows, out_cols = _compute_out_shape(inp_rows, inp_cols, dims)
    kernel = make_reduce_kernel(reduce_fn, inp_rows, inp_cols, dims)

    inp_torch = inp_factory()
    scaler_torch = create_scaler_tile(scaler_val)
    out_shape = (out_rows * TILE, out_cols * TILE)
    out_torch = torch.zeros(out_shape, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    expected_val = _expected_reduce_value(inp_torch, reduce_fn, dims, scaler_val)
    assert result[0, 0].float().item() == pytest.approx(expected_val, rel=0.05, abs=1.0)


# =============================================================================
# Multicore tests: each core reduces its own tile independently.
# =============================================================================


@pytest.mark.parametrize(
    "grid_rows, grid_cols, reduce_fn, dims, test_id",
    [
        (2, 2, "reduce_sum", [0, 1], "sum_scalar_2x2"),
        (2, 2, "reduce_max", [0, 1], "max_scalar_2x2"),
    ],
    ids=["sum_scalar_2x2", "max_scalar_2x2"],
)
def test_reduce_multicore(device, grid_rows, grid_cols, reduce_fn, dims, test_id):
    """Each core in the grid independently reduces its own tile."""
    kernel = make_multicore_reduce_kernel(reduce_fn, grid_rows, grid_cols, dims)

    tensor_rows = grid_rows * TILE
    tensor_cols = grid_cols * TILE
    inp_torch = torch.ones(tensor_rows, tensor_cols, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(tensor_rows, tensor_cols, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    # Each core reduces a single 32x32 all-ones tile. Scalar reduction
    # (dims=[0,1]) gives 1024.0. The result is placed at [0,0] of each
    # core's output tile.
    expected_val = float(TILE * TILE)
    for tile_row in range(grid_rows):
        for tile_col in range(grid_cols):
            actual = result[tile_row * TILE, tile_col * TILE].float().item()
            assert actual == pytest.approx(
                expected_val, rel=0.05
            ), f"core ({tile_row},{tile_col}): got {actual}, expected {expected_val}"
