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

from ttlang_test_utils import assert_allclose, to_l1, to_dram

import ttl

TILE = 32

# =============================================================================
# Kernel generation from templates
# =============================================================================

REDUCE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def reduce_kernel(inp, scaler, out):
    """Reduce {reduce_fn} dims={dims} on ({inp_rows},{inp_cols}) grid."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({inp_rows}, {inp_cols}), block_count=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({out_rows}, {out_cols}), block_count=2
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

@ttl.operation(grid=({grid_cols}, {grid_rows}))
def reduce_kernel(inp, scaler, out):
    """Multicore reduce {reduce_fn} dims={dims}, each core reduces its own tile."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=(1, 1), block_count=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), block_count=2
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
        [0],
        lambda: torch.rand(64, 64, dtype=torch.bfloat16),
        1.0,
        "max_2x2_dim0_random",
    ),
    (
        "reduce_max",
        (2, 2),
        [1],
        lambda: torch.rand(64, 64, dtype=torch.bfloat16),
        1.0,
        "max_2x2_dim1_random",
    ),
    (
        "reduce_max",
        (2, 2),
        [0, 1],
        lambda: torch.rand(64, 64, dtype=torch.bfloat16),
        1.0,
        "max_2x2_both_random",
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
        [0],
        lambda: torch.rand(128, 128, dtype=torch.bfloat16),
        1.0,
        "max_4x4_dim0_random",
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
        "reduce_max",
        (4, 4),
        [0, 1],
        lambda: torch.rand(128, 128, dtype=torch.bfloat16),
        1.0,
        "max_4x4_both_random",
    ),
    # Non-square multi-tile.
    (
        "reduce_sum",
        (2, 1),
        [0],
        lambda: torch.rand(64, 32, dtype=torch.bfloat16),
        1.0,
        "sum_2x1_dim0_random",
    ),
    (
        "reduce_sum",
        (1, 2),
        [1],
        lambda: torch.rand(32, 64, dtype=torch.bfloat16),
        1.0,
        "sum_1x2_dim1_random",
    ),
    (
        "reduce_max",
        (2, 1),
        [0],
        lambda: torch.rand(64, 32, dtype=torch.bfloat16),
        1.0,
        "max_2x1_dim0_random",
    ),
    (
        "reduce_max",
        (1, 2),
        [1],
        lambda: torch.rand(32, 64, dtype=torch.bfloat16),
        1.0,
        "max_1x2_dim1_random",
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
        (2, 2, "reduce_sum", [0, 1], "sum_scalar_2x2_random"),
        (2, 2, "reduce_max", [0, 1], "max_scalar_2x2_random"),
    ],
    ids=[
        "sum_scalar_2x2",
        "max_scalar_2x2",
        "sum_scalar_2x2_random",
        "max_scalar_2x2_random",
    ],
)
def test_reduce_multicore(device, grid_rows, grid_cols, reduce_fn, dims, test_id):
    """Each core in the grid independently reduces its own tile."""
    kernel = make_multicore_reduce_kernel(reduce_fn, grid_rows, grid_cols, dims)

    tensor_rows = grid_rows * TILE
    tensor_cols = grid_cols * TILE
    if "random" in test_id:
        inp_torch = torch.randn(tensor_rows, tensor_cols, dtype=torch.bfloat16)
    else:
        inp_torch = torch.ones(tensor_rows, tensor_cols, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(tensor_rows, tensor_cols, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    for tile_row in range(grid_rows):
        for tile_col in range(grid_cols):
            tile_inp = inp_torch[
                tile_row * TILE : (tile_row + 1) * TILE,
                tile_col * TILE : (tile_col + 1) * TILE,
            ]
            if reduce_fn == "reduce_sum":
                expected_val = tile_inp.float().sum().item()
            else:
                expected_val = tile_inp.float().max().item()

            actual = result[tile_row * TILE, tile_col * TILE].float().item()
            assert actual == pytest.approx(
                expected_val, rel=0.05
            ), f"core ({tile_row},{tile_col}): got {actual}, expected {expected_val}"


# =============================================================================
# L1 accumulation tests: multi-tile reduce_sum with maximize_dst=false.
# Verifies per-tile L1 accumulation (pack_reconfig_l1_acc) works correctly
# when DST accumulation is disabled.
# =============================================================================

# Separate kernel factory to avoid cache collision with DST-accumulation kernels.
_l1_kernel_cache: dict[tuple, Callable] = {}

L1_ACC_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1), options="--no-ttl-maximize-dst")
def reduce_kernel(inp, scaler, out):
    """Reduce {reduce_fn} dims={dims} with L1 accumulation."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({inp_rows}, {inp_cols}), block_count=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({out_rows}, {out_cols}), block_count=2
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


def _make_l1_acc_kernel(
    reduce_fn: str, inp_rows: int, inp_cols: int, dims: List[int]
) -> Callable:
    out_rows, out_cols = _compute_out_shape(inp_rows, inp_cols, dims)
    cache_key = (reduce_fn, inp_rows, inp_cols, tuple(dims))
    if cache_key in _l1_kernel_cache:
        return _l1_kernel_cache[cache_key]

    code = L1_ACC_TEMPLATE.format(
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
        mode="w", suffix=".py", delete=False, prefix=f"l1acc_{reduce_fn}_"
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("l1_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.reduce_kernel
    _l1_kernel_cache[cache_key] = kernel
    return kernel


@pytest.mark.parametrize(
    "reduce_fn, inp_shape, dims, inp_factory, scaler_val, test_id",
    [
        (
            "reduce_sum",
            (2, 2),
            [0],
            lambda: torch.rand(64, 64, dtype=torch.bfloat16),
            1.0,
            "l1_sum_2x2_dim0",
        ),
        (
            "reduce_sum",
            (2, 2),
            [1],
            lambda: torch.rand(64, 64, dtype=torch.bfloat16),
            1.0,
            "l1_sum_2x2_dim1",
        ),
        (
            "reduce_sum",
            (2, 2),
            [0, 1],
            lambda: torch.rand(64, 64, dtype=torch.bfloat16),
            1.0,
            "l1_sum_2x2_both",
        ),
        (
            "reduce_sum",
            (4, 4),
            [0],
            lambda: torch.rand(128, 128, dtype=torch.bfloat16),
            1.0,
            "l1_sum_4x4_dim0",
        ),
    ],
    ids=["l1_sum_2x2_dim0", "l1_sum_2x2_dim1", "l1_sum_2x2_both", "l1_sum_4x4_dim0"],
)
def test_reduce_l1_accumulation(
    device, reduce_fn, inp_shape, dims, inp_factory, scaler_val, test_id
):
    """Multi-tile reduce_sum with L1 accumulation (maximize_dst=false)."""
    inp_rows, inp_cols = inp_shape
    out_rows, out_cols = _compute_out_shape(inp_rows, inp_cols, dims)
    kernel = _make_l1_acc_kernel(reduce_fn, inp_rows, inp_cols, dims)

    inp_torch = inp_factory()
    scaler_torch = create_scaler_tile(scaler_val)
    out_shape = (out_rows * TILE, out_cols * TILE)
    out_torch = torch.zeros(*out_shape, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    expected_val = _expected_reduce_value(inp_torch, reduce_fn, dims, scaler_val)
    assert result[0, 0].float().item() == pytest.approx(expected_val, rel=0.05, abs=1.0)


# =============================================================================
# Reduce -> broadcast chaining tests.
# =============================================================================

REDUCE_BCAST_TEMPLATE = """
import ttl

@ttl.operation(grid=(1, 1))
def reduce_bcast_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({inp_rows}, {inp_cols}), block_count=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), block_count=2
    )
    reduced_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=({red_rows}, {red_cols}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({inp_rows}, {inp_cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, scaler_dfb.wait() as s:
            with reduced_dfb.reserve() as r:
                r.store(ttl.math.{reduce_fn}(x, s, dims={dims}))
            with reduced_dfb.wait() as r, out_dfb.reserve() as o:
                o.store(ttl.math.broadcast(r, o, dims={dims}))

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[{inp_slice}], blk).wait()
        blk.push()
        sblk = scaler_dfb.reserve()
        ttl.copy(scaler[0, 0], sblk).wait()
        sblk.push()

    @ttl.datamovement()
    def dm_write():
        blk = out_dfb.wait()
        ttl.copy(blk, out[{out_slice}]).wait()
        blk.pop()
"""

_reduce_bcast_cache: dict[tuple, Callable] = {}


def _make_reduce_bcast_kernel(
    reduce_fn: str, inp_rows: int, inp_cols: int, dims: List[int]
) -> Callable:
    red_rows = 1 if 0 in dims else inp_rows
    red_cols = 1 if 1 in dims else inp_cols
    cache_key = ("reduce_bcast", reduce_fn, inp_rows, inp_cols, tuple(dims))
    if cache_key in _reduce_bcast_cache:
        return _reduce_bcast_cache[cache_key]

    code = REDUCE_BCAST_TEMPLATE.format(
        reduce_fn=reduce_fn,
        inp_rows=inp_rows,
        inp_cols=inp_cols,
        red_rows=red_rows,
        red_cols=red_cols,
        dims=dims,
        inp_slice=_slice_syntax(inp_rows, inp_cols),
        out_slice=_slice_syntax(inp_rows, inp_cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"reduce_bcast_{reduce_fn}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("reduce_bcast_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.reduce_bcast_kernel
    _reduce_bcast_cache[cache_key] = kernel
    return kernel


@pytest.mark.parametrize(
    "reduce_fn, inp_shape, dims, test_id",
    [
        ("reduce_sum", (2, 2), [0, 1], "sum_2x2_scalar"),
        ("reduce_sum", (2, 2), [0], "sum_2x2_dim0"),
        ("reduce_sum", (2, 2), [1], "sum_2x2_dim1"),
        ("reduce_max", (2, 2), [0, 1], "max_2x2_scalar"),
        ("reduce_max", (2, 2), [0], "max_2x2_dim0"),
        ("reduce_max", (2, 2), [1], "max_2x2_dim1"),
        ("reduce_sum", (4, 4), [0], "sum_4x4_dim0"),
        ("reduce_sum", (4, 4), [1], "sum_4x4_dim1"),
    ],
    ids=[
        "sum_2x2_scalar",
        "sum_2x2_dim0",
        "sum_2x2_dim1",
        "max_2x2_scalar",
        "max_2x2_dim0",
        "max_2x2_dim1",
        "sum_4x4_dim0",
        "sum_4x4_dim1",
    ],
)
def test_reduce_broadcast_chain(device, reduce_fn, inp_shape, dims, test_id):
    """Reduce then broadcast back to original shape."""
    inp_rows, inp_cols = inp_shape
    kernel = _make_reduce_bcast_kernel(reduce_fn, inp_rows, inp_cols, dims)

    inp_torch = torch.rand(inp_rows * TILE, inp_cols * TILE, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(inp_rows * TILE, inp_cols * TILE, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    inp_f = inp_torch.float()
    if reduce_fn == "reduce_sum":
        reduced = inp_f.sum(dim=dims, keepdim=True)
    else:
        reduced = inp_f.amax(dim=dims, keepdim=True)
    expected = reduced.expand_as(inp_f)

    from utils.correctness import assert_allclose

    assert_allclose(result.float(), expected, rtol=0.1, atol=1.0)


# =============================================================================
# DRAM memory config tests.
# =============================================================================


@pytest.mark.parametrize(
    "reduce_fn, dims, test_id",
    [
        ("reduce_sum", [0, 1], "sum_dram"),
        ("reduce_max", [0, 1], "max_dram"),
    ],
    ids=["sum_dram", "max_dram"],
)
def test_reduce_dram(device, reduce_fn, dims, test_id):
    """Single-tile reduce with DRAM-interleaved tensors."""
    kernel = make_reduce_kernel(reduce_fn, 1, 1, dims)

    inp_torch = torch.rand(TILE, TILE, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    scaler = to_dram(scaler_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    expected_val = _expected_reduce_value(inp_torch, reduce_fn, dims, 1.0)
    assert result[0, 0].float().item() == pytest.approx(expected_val, rel=0.05, abs=1.0)


# =============================================================================
# Multicore row/col reduce tests.
# =============================================================================

MULTICORE_ROW_COL_TEMPLATE = """
import ttl

@ttl.operation(grid=({grid_rows}, {grid_cols}))
def reduce_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=(1, {inp_cols}), block_count=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, {out_cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, scaler_dfb.wait() as scaler_blk:
            with out_dfb.reserve() as out_blk:
                result = ttl.math.{reduce_fn}(inp_blk, scaler_blk, dims={dims})
                out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        inp_blk = inp_dfb.reserve()
        ttl.copy(inp[y, x * {inp_cols} : x * {inp_cols} + {inp_cols}], inp_blk).wait()
        inp_blk.push()
        sblk = scaler_dfb.reserve()
        ttl.copy(scaler[0, 0], sblk).wait()
        sblk.push()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        blk = out_dfb.wait()
        ttl.copy(blk, out[y, x * {out_cols} : x * {out_cols} + {out_cols}]).wait()
        blk.pop()
"""

_multicore_rc_cache: dict[tuple, Callable] = {}


def _make_multicore_row_col_kernel(
    reduce_fn: str,
    grid_rows: int,
    grid_cols: int,
    inp_cols: int,
    out_cols: int,
    dims: List[int],
) -> Callable:
    cache_key = (
        "mc_rc",
        reduce_fn,
        grid_rows,
        grid_cols,
        inp_cols,
        out_cols,
        tuple(dims),
    )
    if cache_key in _multicore_rc_cache:
        return _multicore_rc_cache[cache_key]

    code = MULTICORE_ROW_COL_TEMPLATE.format(
        reduce_fn=reduce_fn,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        inp_cols=inp_cols,
        out_cols=out_cols,
        dims=dims,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"mc_rc_{reduce_fn}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("mc_rc_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.reduce_kernel
    _multicore_rc_cache[cache_key] = kernel
    return kernel


@pytest.mark.parametrize(
    "reduce_fn, grid_rows, grid_cols, inp_cols, dims, test_id",
    [
        ("reduce_sum", 1, 2, 2, [1], "sum_col_1x2"),
        ("reduce_sum", 2, 1, 1, [0, 1], "sum_scalar_multicore_distinct"),
    ],
    ids=["sum_col_1x2", "sum_scalar_multicore_distinct"],
)
def test_reduce_multicore_row_col(
    device, reduce_fn, grid_rows, grid_cols, inp_cols, dims, test_id
):
    """Multicore reduce with row/col dimensions."""
    out_cols = 1 if 1 in dims else inp_cols
    kernel = _make_multicore_row_col_kernel(
        reduce_fn, grid_rows, grid_cols, inp_cols, out_cols, dims
    )

    total_rows = grid_rows * TILE
    total_cols = grid_cols * inp_cols * TILE
    inp_torch = torch.rand(total_rows, total_cols, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_total_cols = grid_cols * out_cols * TILE
    out_torch = torch.zeros(total_rows, out_total_cols, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    # Verify first core's output.
    from utils.correctness import assert_allclose

    core_inp = inp_torch[:TILE, : inp_cols * TILE].float()
    if reduce_fn == "reduce_sum":
        expected = core_inp.sum(dim=dims, keepdim=True)
    else:
        expected = core_inp.amax(dim=dims, keepdim=True)
    actual = result[:TILE, : out_cols * TILE].float()
    assert_allclose(actual[0, 0], expected.flatten()[0], rtol=0.05, atol=1.0)


# =============================================================================
# Multicore + multitile tests.
# =============================================================================

MULTICORE_MULTITILE_TEMPLATE = """
import ttl

@ttl.operation(grid=({grid_rows}, {grid_cols}))
def reduce_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({tile_rows}, {tile_cols}), block_count=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({out_tile_rows}, {out_tile_cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, scaler_dfb.wait() as scaler_blk:
            with out_dfb.reserve() as out_blk:
                out_blk.store(ttl.math.{reduce_fn}(inp_blk, scaler_blk, dims={dims}))

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        inp_blk = inp_dfb.reserve()
        ttl.copy(inp[y * {tile_rows} : y * {tile_rows} + {tile_rows},
                     x * {tile_cols} : x * {tile_cols} + {tile_cols}], inp_blk).wait()
        inp_blk.push()
        sblk = scaler_dfb.reserve()
        ttl.copy(scaler[0, 0], sblk).wait()
        sblk.push()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        blk = out_dfb.wait()
        ttl.copy(blk, out[y * {out_tile_rows} : y * {out_tile_rows} + {out_tile_rows},
                         x * {out_tile_cols} : x * {out_tile_cols} + {out_tile_cols}]).wait()
        blk.pop()
"""

_mc_mt_cache: dict[tuple, Callable] = {}


def _make_mc_mt_kernel(
    reduce_fn: str,
    grid_rows: int,
    grid_cols: int,
    tile_rows: int,
    tile_cols: int,
    dims: List[int],
) -> Callable:
    out_tile_rows = 1 if 0 in dims else tile_rows
    out_tile_cols = 1 if 1 in dims else tile_cols
    cache_key = (
        "mc_mt",
        reduce_fn,
        grid_rows,
        grid_cols,
        tile_rows,
        tile_cols,
        tuple(dims),
    )
    if cache_key in _mc_mt_cache:
        return _mc_mt_cache[cache_key]

    code = MULTICORE_MULTITILE_TEMPLATE.format(
        reduce_fn=reduce_fn,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        out_tile_rows=out_tile_rows,
        out_tile_cols=out_tile_cols,
        dims=dims,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"mc_mt_{reduce_fn}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("mc_mt_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.reduce_kernel
    _mc_mt_cache[cache_key] = kernel
    return kernel


@pytest.mark.parametrize(
    "reduce_fn, grid, tiles, dims, test_id",
    [
        ("reduce_sum", (2, 2), (2, 2), [0, 1], "sum_2x2_grid_2x2_tiles_scalar"),
        ("reduce_sum", (2, 2), (2, 2), [0], "sum_2x2_grid_2x2_tiles_dim0"),
        ("reduce_max", (2, 2), (2, 2), [0, 1], "max_2x2_grid_2x2_tiles_scalar"),
    ],
    ids=[
        "sum_2x2_grid_2x2_tiles_scalar",
        "sum_2x2_grid_2x2_tiles_dim0",
        "max_2x2_grid_2x2_tiles_scalar",
    ],
)
def test_reduce_multicore_multitile(device, reduce_fn, grid, tiles, dims, test_id):
    """Multicore with multi-tile blocks per core."""
    grid_rows, grid_cols = grid
    tile_rows, tile_cols = tiles
    kernel = _make_mc_mt_kernel(
        reduce_fn, grid_rows, grid_cols, tile_rows, tile_cols, dims
    )

    total_rows = grid_rows * tile_rows * TILE
    total_cols = grid_cols * tile_cols * TILE
    inp_torch = torch.rand(total_rows, total_cols, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)

    out_tile_rows = 1 if 0 in dims else tile_rows
    out_tile_cols = 1 if 1 in dims else tile_cols
    out_rows = grid_rows * out_tile_rows * TILE
    out_cols = grid_cols * out_tile_cols * TILE
    out_torch = torch.zeros(out_rows, out_cols, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    # Check first core's output.
    core_inp = inp_torch[: tile_rows * TILE, : tile_cols * TILE].float()
    if reduce_fn == "reduce_sum":
        expected = core_inp.sum(dim=dims, keepdim=True)
    else:
        expected = core_inp.amax(dim=dims, keepdim=True)

    from utils.correctness import assert_allclose

    assert_allclose(result[0, 0].float(), expected.flatten()[0], rtol=0.05, atol=1.0)


# =============================================================================
# Composition: broadcast -> reduce (multicore multitile).
# =============================================================================

BCAST_REDUCE_TEMPLATE = """
import ttl

@ttl.operation(grid=({grid_rows}, {grid_cols}))
def bcast_reduce_kernel(inp, bcast_in, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({tile_rows}, {tile_cols}), block_count=2
    )
    bcast_dfb = ttl.make_dataflow_buffer_like(
        bcast_in, shape=(1, 1), block_count=2
    )
    bcast_out_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({tile_rows}, {tile_cols}), block_count=2
    )
    scaler_dfb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), block_count=2
    )
    add_out_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({tile_rows}, {tile_cols}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with bcast_dfb.wait() as b, bcast_out_dfb.reserve() as bo:
            bo.store(ttl.math.broadcast(b, bo, dims=[0, 1]))

        with inp_dfb.wait() as x, bcast_out_dfb.wait() as bv, add_out_dfb.reserve() as ao:
            ao.store(x + bv)

        with add_out_dfb.wait() as av, scaler_dfb.wait() as s, out_dfb.reserve() as o:
            o.store(ttl.math.reduce_sum(av, s, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        blk = inp_dfb.reserve()
        ttl.copy(inp[y * {tile_rows} : y * {tile_rows} + {tile_rows},
                     x * {tile_cols} : x * {tile_cols} + {tile_cols}], blk).wait()
        blk.push()
        bblk = bcast_dfb.reserve()
        ttl.copy(bcast_in[0, 0], bblk).wait()
        bblk.push()
        sblk = scaler_dfb.reserve()
        ttl.copy(scaler[0, 0], sblk).wait()
        sblk.push()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        blk = out_dfb.wait()
        ttl.copy(blk, out[y, x]).wait()
        blk.pop()
"""


def test_bcast_then_reduce_multicore_multitile(device):
    """Broadcast scalar, add to input, then reduce. 2x2 grid, 2x2 tiles each."""
    grid_rows, grid_cols = 2, 2
    tile_rows, tile_cols = 2, 2

    code = BCAST_REDUCE_TEMPLATE.format(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="bcast_reduce_"
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name
    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("bcast_reduce_mod", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.bcast_reduce_kernel

    total_rows = grid_rows * tile_rows * TILE
    total_cols = grid_cols * tile_cols * TILE
    inp_torch = torch.ones(total_rows, total_cols, dtype=torch.bfloat16)
    bcast_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)
    bcast_torch[0, 0] = 2.0
    bcast_torch[16, 0] = 2.0
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(grid_rows * TILE, grid_cols * TILE, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    bcast_in = to_l1(bcast_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, bcast_in, scaler, out)
    result = ttnn.to_torch(out)

    # Each core: sum((1.0 + 2.0) * 2*2 tiles * 32*32 elements) = 3.0 * 4096 = 12288
    expected_val = 3.0 * tile_rows * tile_cols * TILE * TILE
    actual = result[0, 0].float().item()
    assert actual == pytest.approx(
        expected_val, rel=0.05
    ), f"core (0,0): got {actual}, expected {expected_val}"


# =============================================================================
# Composition: matmul -> reduce, reduce -> bcast -> matmul.
# =============================================================================


@ttl.operation(grid=(1, 1))
def matmul_then_reduce_kernel(mat_a, mat_b, scaler, out):
    a_dfb = ttl.make_dataflow_buffer_like(mat_a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(mat_b, shape=(1, 1), block_count=2)
    mm_dfb = ttl.make_dataflow_buffer_like(mat_a, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a, b_dfb.wait() as b, mm_dfb.reserve() as mm:
            mm.store(a @ b)
        with mm_dfb.wait() as mm, scaler_dfb.wait() as s, out_dfb.reserve() as o:
            o.store(ttl.math.reduce_sum(mm, s, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        ablk = a_dfb.reserve()
        ttl.copy(mat_a[0, 0], ablk).wait()
        ablk.push()
        bblk = b_dfb.reserve()
        ttl.copy(mat_b[0, 0], bblk).wait()
        bblk.push()
        sblk = scaler_dfb.reserve()
        ttl.copy(scaler[0, 0], sblk).wait()
        sblk.push()

    @ttl.datamovement()
    def dm_write():
        blk = out_dfb.wait()
        ttl.copy(blk, out[0, 0]).wait()
        blk.pop()


def test_matmul_then_reduce(device):
    """matmul(A, B) -> reduce_sum: single tile."""
    a_torch = torch.rand(TILE, TILE, dtype=torch.bfloat16)
    b_torch = torch.rand(TILE, TILE, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    matmul_then_reduce_kernel(a, b, scaler, out)
    result = ttnn.to_torch(out)

    expected = (a_torch.float() @ b_torch.float()).sum().item()
    actual = result[0, 0].float().item()
    assert actual == pytest.approx(
        expected, rel=0.1, abs=10.0
    ), f"got {actual}, expected {expected}"


# =============================================================================
# Composition: reduce -> broadcast -> matmul (with K-loop accumulation).
# =============================================================================

REDUCE_BCAST_MATMUL_TEMPLATE = """
import ttl

@ttl.operation(grid=(1, 1), fp32_dest_acc_en=True)
def reduce_bcast_matmul_kernel(reduce_in, mat_b, scaler, out):
    reduce_dfb = ttl.make_dataflow_buffer_like(reduce_in, shape=({mt}, {kt}), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(scaler, shape=({mt}, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(mat_b, shape=(1, {nt}), block_count=2)
    partial_dfb = ttl.make_dataflow_buffer_like(out, shape=({mt}, {nt}), block_count=2)
    acc_dfb = ttl.make_dataflow_buffer_like(out, shape=({mt}, {nt}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({mt}, {nt}), block_count=2)

    @ttl.compute()
    def compute_fn():
        with reduce_dfb.wait() as r_in, sc_dfb.wait() as s:
            with red_dfb.reserve() as r_out:
                r_out.store(ttl.math.reduce_sum(r_in, s, dims=[0, 1]))
        with red_dfb.wait() as r_val, bcast_dfb.reserve() as b_out:
            b_out.store(ttl.math.broadcast(r_val, b_out, dims=[0, 1]))

        bcast_blk = bcast_dfb.wait()

        # First K iteration: matmul to accumulator
        with b_dfb.wait() as b_blk, acc_dfb.reserve() as acc:
            acc.store(bcast_blk @ b_blk)

        # Remaining K iterations: matmul to partial, add to accumulator.
        for _ in range({kt} - 1):
            with b_dfb.wait() as b_blk, partial_dfb.reserve() as p:
                p.store(bcast_blk @ b_blk)
            with partial_dfb.wait() as new_val, acc_dfb.wait() as prev:
                with acc_dfb.reserve() as acc:
                    acc.store(prev + new_val)

        bcast_blk.pop()

        # Copy accumulator to output.
        with acc_dfb.wait() as final_val, out_dfb.reserve() as o:
            o.store(final_val)

    @ttl.datamovement()
    def dm_read():
        rblk = reduce_dfb.reserve()
        ttl.copy(reduce_in[0:{mt}, 0:{kt}], rblk).wait()
        rblk.push()
        sblk = sc_dfb.reserve()
        ttl.copy(scaler[0, 0], sblk).wait()
        sblk.push()
        for kt_idx in range({kt}):
            bblk = b_dfb.reserve()
            ttl.copy(mat_b[kt_idx, 0:{nt}], bblk).wait()
            bblk.push()

    @ttl.datamovement()
    def dm_write():
        blk = out_dfb.wait()
        ttl.copy(blk, out[0:{mt}, 0:{nt}]).wait()
        blk.pop()
"""


@pytest.mark.parametrize(
    "mt, kt, nt, test_id",
    [
        (1, 1, 1, "1x1x1"),
        (2, 1, 2, "2x1x2"),
        (2, 2, 2, "2x2x2"),
        (2, 4, 2, "2x4x2"),
        (4, 1, 2, "4x1x2"),
    ],
    ids=["1x1x1", "2x1x2", "2x2x2", "2x4x2", "4x1x2"],
)
def test_reduce_bcast_matmul(device, mt, kt, nt, test_id):
    """reduce_sum -> scalar broadcast(Mt,1) -> K-loop matmul with accumulation.

    TODO: simplify once c += a @ b accumulation is supported.
    """
    size_m = mt * TILE
    size_k = kt * TILE
    size_n = nt * TILE

    code = REDUCE_BCAST_MATMUL_TEMPLATE.format(mt=mt, kt=kt, nt=nt)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="rbm_"
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name
    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("rbm_mod", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.reduce_bcast_matmul_kernel

    torch.manual_seed(12345)
    r_torch = torch.randn(size_m, size_k, dtype=torch.bfloat16)
    b_torch = torch.randn(size_k, size_n, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(size_m, size_n, dtype=torch.bfloat16)

    r = to_l1(r_torch, device)
    b = to_l1(b_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(r, b, scaler, out)
    result = ttnn.to_torch(out)

    scalar_val = r_torch.float().sum().item()
    bcast_mat = torch.full((size_m, size_k), scalar_val, dtype=torch.float32)
    expected = bcast_mat @ b_torch.float()

    from utils.correctness import assert_pcc

    assert_pcc(expected, result.float(), threshold=0.99)


# =============================================================================
# Reduce -> broadcast type coverage.
# =============================================================================

REDUCE_BCAST_TYPE_TEMPLATE = """
import ttl

@ttl.operation(grid=(1, 1))
def reduce_bcast_type_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({inp_rows}, {inp_cols}), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({out_rows}, {out_cols}), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, sc_dfb.wait() as s:
            with red_dfb.reserve() as r:
                r.store(ttl.math.reduce_sum(x, s, dims=[0, 1]))
            with red_dfb.wait() as r, out_dfb.reserve() as o:
                o.store(ttl.math.broadcast(r, o, dims={bcast_dims}))

    @ttl.datamovement()
    def dm_read():
        blk = inp_dfb.reserve()
        ttl.copy(inp[{inp_slice}], blk).wait()
        blk.push()
        sblk = sc_dfb.reserve()
        ttl.copy(scaler[0, 0], sblk).wait()
        sblk.push()

    @ttl.datamovement()
    def dm_write():
        blk = out_dfb.wait()
        ttl.copy(blk, out[{out_slice}]).wait()
        blk.pop()
"""


@pytest.mark.parametrize(
    "inp_shape, bcast_dims, out_shape, test_id",
    [
        ((1, 1), [0, 1], (1, 1), "scalar_1x1_to_1x1"),
        ((1, 1), [0, 1], (2, 2), "scalar_1x1_to_2x2"),
        ((1, 1), [0, 1], (2, 1), "scalar_1x1_to_2x1"),
        ((1, 1), [0, 1], (1, 2), "scalar_1x1_to_1x2"),
        ((2, 2), [0, 1], (2, 1), "scalar_2x2_to_2x1"),
        ((2, 2), [0, 1], (1, 2), "scalar_2x2_to_1x2"),
        ((2, 2), [0, 1], (2, 1), "scalar_2x2_to_2x1_bcast_row"),
        ((2, 2), [0, 1], (1, 2), "scalar_2x2_to_1x2_bcast_col"),
    ],
    ids=[
        "scalar_1x1_to_1x1",
        "scalar_1x1_to_2x2",
        "scalar_1x1_to_2x1",
        "scalar_1x1_to_1x2",
        "scalar_2x2_to_2x1",
        "scalar_2x2_to_1x2",
        "scalar_2x2_to_2x1_bcast_row",
        "scalar_2x2_to_1x2_bcast_col",
    ],
)
def test_reduce_bcast_type(device, inp_shape, bcast_dims, out_shape, test_id):
    """Reduce to scalar then broadcast with specific type."""
    inp_rows, inp_cols = inp_shape
    out_rows, out_cols = out_shape

    code = REDUCE_BCAST_TYPE_TEMPLATE.format(
        inp_rows=inp_rows,
        inp_cols=inp_cols,
        out_rows=out_rows,
        out_cols=out_cols,
        bcast_dims=bcast_dims,
        inp_slice=_slice_syntax(inp_rows, inp_cols),
        out_slice=_slice_syntax(out_rows, out_cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="rbt_"
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name
    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("rbt_mod", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.reduce_bcast_type_kernel

    inp_torch = torch.rand(inp_rows * TILE, inp_cols * TILE, dtype=torch.bfloat16)
    scaler_torch = create_scaler_tile(1.0)
    out_torch = torch.zeros(out_rows * TILE, out_cols * TILE, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, scaler, out)
    result = ttnn.to_torch(out)

    scalar_val = inp_torch.float().sum().item()
    expected = torch.full_like(result.float(), scalar_val)

    from utils.correctness import assert_allclose

    assert_allclose(result.float(), expected, rtol=0.01, atol=10.0)
