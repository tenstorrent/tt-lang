# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTL transpose operation.

Uses dynamically generated kernels to parameterize over block shapes and
input data patterns. Covers single-tile, multi-tile non-square grids,
and double-transpose identity.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import atexit
import importlib
import os
import tempfile
from typing import Callable

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

TILE = 32

# =============================================================================
# Kernel generation
# =============================================================================

TRANSPOSE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def transpose_kernel(inp, out):
    """Transpose ({inp_rows},{inp_cols}) -> ({out_rows},{out_cols})."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({inp_rows}, {inp_cols}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({out_rows}, {out_cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            result = ttl.math.transpose(inp_blk)
            out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[{inp_slice}], inp_blk)
        tx_inp.wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[{out_slice}])
        tx_out.wait()
        out_blk.pop()
'''

DOUBLE_TRANSPOSE_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def double_transpose_kernel(inp, out):
    """Transpose twice: ({rows},{cols}) -> ({cols},{rows}) -> ({rows},{cols})."""
    inp_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({rows}, {cols}), block_count=2
    )
    mid_dfb = ttl.make_dataflow_buffer_like(
        inp, shape=({cols}, {rows}), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=({rows}, {cols}), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, mid_dfb.reserve() as mid_blk:
            result = ttl.math.transpose(inp_blk)
            mid_blk.store(result)
        with mid_dfb.wait() as mid_blk2, out_dfb.reserve() as out_blk:
            result2 = ttl.math.transpose(mid_blk2)
            out_blk.store(result2)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[{slice}], inp_blk)
        tx_inp.wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[{slice}])
        tx_out.wait()
        out_blk.pop()
'''

_kernel_cache = {}
_temp_files = []


def _slice_syntax(rows: int, cols: int) -> str:
    if rows == 1 and cols == 1:
        return "0, 0"
    return f"0:{rows}, 0:{cols}"


def make_transpose_kernel(inp_rows: int, inp_cols: int) -> Callable:
    """Generate a transpose kernel for the given input grid shape."""
    out_rows, out_cols = inp_cols, inp_rows
    cache_key = ("transpose", inp_rows, inp_cols)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    code = TRANSPOSE_KERNEL_TEMPLATE.format(
        inp_rows=inp_rows,
        inp_cols=inp_cols,
        out_rows=out_rows,
        out_cols=out_cols,
        inp_slice=_slice_syntax(inp_rows, inp_cols),
        out_slice=_slice_syntax(out_rows, out_cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"transpose_{inp_rows}x{inp_cols}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("transpose_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.transpose_kernel
    _kernel_cache[cache_key] = kernel
    return kernel


def make_double_transpose_kernel(rows: int, cols: int) -> Callable:
    """Generate a double-transpose kernel."""
    cache_key = ("double_transpose", rows, cols)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    code = DOUBLE_TRANSPOSE_KERNEL_TEMPLATE.format(
        rows=rows,
        cols=cols,
        slice=_slice_syntax(rows, cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"double_transpose_{rows}x{cols}_",
    ) as tmp:
        tmp.write(code)
        temp_path = tmp.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("double_transpose_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.double_transpose_kernel
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
# Test configurations: (inp_rows, inp_cols, inp_factory, description)
# =============================================================================

TRANSPOSE_CONFIGS = [
    # Single-tile
    (
        1,
        1,
        lambda r, c: torch.ones(r * TILE, c * TILE, dtype=torch.bfloat16),
        "1x1_ones",
    ),
    (
        1,
        1,
        lambda r, c: torch.arange(r * TILE, dtype=torch.bfloat16)
        .unsqueeze(1)
        .expand(r * TILE, c * TILE)
        .contiguous(),
        "1x1_row_values",
    ),
    (
        1,
        1,
        lambda r, c: torch.arange(c * TILE, dtype=torch.bfloat16)
        .unsqueeze(0)
        .expand(r * TILE, c * TILE)
        .contiguous(),
        "1x1_col_values",
    ),
    (
        1,
        1,
        lambda r, c: torch.arange(r * c * TILE * TILE, dtype=torch.bfloat16).reshape(
            r * TILE, c * TILE
        ),
        "1x1_unique",
    ),
    # Non-square multi-tile (verifies grid reindexing)
    (
        2,
        3,
        lambda r, c: torch.arange(r * c * TILE * TILE, dtype=torch.bfloat16).reshape(
            r * TILE, c * TILE
        ),
        "2x3_unique",
    ),
    (
        1,
        3,
        lambda r, c: torch.arange(r * c * TILE * TILE, dtype=torch.bfloat16).reshape(
            r * TILE, c * TILE
        ),
        "1x3_unique",
    ),
    (
        3,
        1,
        lambda r, c: torch.arange(r * c * TILE * TILE, dtype=torch.bfloat16).reshape(
            r * TILE, c * TILE
        ),
        "3x1_unique",
    ),
    # More non-square shapes.
    (
        4,
        2,
        lambda r, c: torch.arange(r * c * TILE * TILE, dtype=torch.bfloat16).reshape(
            r * TILE, c * TILE
        ),
        "4x2_unique",
    ),
    (
        2,
        4,
        lambda r, c: torch.arange(r * c * TILE * TILE, dtype=torch.bfloat16).reshape(
            r * TILE, c * TILE
        ),
        "2x4_unique",
    ),
    # Random data.
    (
        1,
        1,
        lambda r, c: torch.rand(r * TILE, c * TILE, dtype=torch.bfloat16),
        "1x1_random",
    ),
    (
        2,
        3,
        lambda r, c: torch.rand(r * TILE, c * TILE, dtype=torch.bfloat16),
        "2x3_random",
    ),
    (
        3,
        2,
        lambda r, c: torch.rand(r * TILE, c * TILE, dtype=torch.bfloat16),
        "3x2_random",
    ),
    (
        4,
        2,
        lambda r, c: torch.rand(r * TILE, c * TILE, dtype=torch.bfloat16),
        "4x2_random",
    ),
]


@pytest.mark.parametrize(
    "inp_rows, inp_cols, inp_factory, test_id",
    TRANSPOSE_CONFIGS,
    ids=[cfg[-1] for cfg in TRANSPOSE_CONFIGS],
)
def test_transpose(device, inp_rows, inp_cols, inp_factory, test_id):
    """Transpose with parameterized shapes and data patterns."""
    kernel = make_transpose_kernel(inp_rows, inp_cols)

    inp_torch = inp_factory(inp_rows, inp_cols)
    out_rows, out_cols = inp_cols, inp_rows
    out_torch = torch.zeros(out_rows * TILE, out_cols * TILE, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    expected = inp_torch.T
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# Double transpose (identity property)
DOUBLE_TRANSPOSE_SHAPES = [(1, 1), (2, 2)]


@pytest.mark.parametrize(
    "rows, cols",
    DOUBLE_TRANSPOSE_SHAPES,
    ids=[f"{r}x{c}" for r, c in DOUBLE_TRANSPOSE_SHAPES],
)
def test_double_transpose(device, rows, cols):
    """Transpose twice should produce the original tensor."""
    kernel = make_double_transpose_kernel(rows, cols)

    inp_torch = torch.arange(rows * cols * TILE * TILE, dtype=torch.bfloat16).reshape(
        rows * TILE, cols * TILE
    )
    out_torch = torch.zeros_like(inp_torch)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), inp_torch.float(), rtol=1e-2, atol=1e-2)
