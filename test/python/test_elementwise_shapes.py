# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTL elementwise operations across multiple tensor shapes.

Tests elementwise ops against PyTorch equivalents with L1 memory configuration,
sweeping 1D tile configurations (row or column vectors only).

Parameterized over:
- Operations: binary (add, sub, mul, max) and unary (exp, log, sqrt, etc.)
- Shapes: 1D tile configurations (1x1, 1x2, 1x3, 2x1, 3x1) - 5 shapes total

Total test cases: 13 ops x 5 shapes = 65 tests
"""

# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import atexit
import importlib.util
import os
import tempfile
from typing import Callable

import pytest
import ttnn
import torch
from test_helpers import assert_allclose, to_dram

pytestmark = pytest.mark.requires_ttnn

# =============================================================================
# Shape Configurations - 1D tile configurations (row or column vectors)
# =============================================================================

TILE_SIZE = 32

TILE_SHAPES = [(r, c) for r in range(1, 5) for c in range(1, 5)]


def tiles_to_tensor_shape(tile_rows: int, tile_cols: int) -> tuple[int, int]:
    """Convert tile grid dimensions to tensor shape."""
    return (tile_rows * TILE_SIZE, tile_cols * TILE_SIZE)


# =============================================================================
# Kernel Templates - parameterized by shape
# =============================================================================

BINARY_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel for {tile_rows}x{tile_cols} tiles."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    out_cb = ttl.make_circular_buffer_like(out, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})

    @ttl.compute()
    def compute_fn():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l {op} r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[{slice_syntax}], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[{slice_syntax}], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[{slice_syntax}])
        tx.wait()
        out_cb.pop()

'''

BINARY_FN_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel (function call) for {tile_rows}x{tile_cols} tiles."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    out_cb = ttl.make_circular_buffer_like(out, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})

    @ttl.compute()
    def compute_fn():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = ttl.math.{op}(l, r)
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[{slice_syntax}], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[{slice_syntax}], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[{slice_syntax}])
        tx.wait()
        out_cb.pop()

'''

UNARY_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def {name}_kernel(inp, out):
    """Unary {name} kernel for {tile_rows}x{tile_cols} tiles."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    out_cb = ttl.make_circular_buffer_like(out, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = ttl.math.{op}(x)
        o.store(result)
        inp_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx_inp = ttl.copy(inp[{slice_syntax}], inp_blk)
        tx_inp.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[{slice_syntax}])
        tx.wait()
        out_cb.pop()

'''


# =============================================================================
# Kernel Factories - generate kernels with specific shapes
# =============================================================================


def _get_slice_syntax(tile_rows: int, tile_cols: int) -> str:
    """Return slice syntax for tensor indexing based on tile shape."""
    if tile_rows == 1 and tile_cols == 1:
        return "0, 0"
    return f"0:{tile_rows}, 0:{tile_cols}"


# Cache for generated kernels: (name, op, tile_rows, tile_cols) -> kernel
_kernel_cache: dict[tuple, Callable] = {}

# Track temp files for cleanup
_temp_files: list[str] = []


def _cleanup_temp_files() -> None:
    """Clean up all temporary kernel files at exit."""
    for path in _temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_cleanup_temp_files)


def make_binary_kernel(name: str, op: str, tile_rows: int, tile_cols: int) -> Callable:
    """Generate a binary kernel for the given shape."""
    cache_key = (name, op, tile_rows, tile_cols, "binary")
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    buffer_factor = 2
    code = BINARY_KERNEL_TEMPLATE.format(
        name=name,
        op=op,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        buffer_factor=buffer_factor,
        slice_syntax=_get_slice_syntax(tile_rows, tile_cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{name}_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = getattr(module, f"{name}_kernel")
    _kernel_cache[cache_key] = kernel
    return kernel


def make_binary_fn_kernel(
    name: str, op: str, tile_rows: int, tile_cols: int
) -> Callable:
    """Generate a binary kernel using function call syntax."""
    cache_key = (name, op, tile_rows, tile_cols, "binary_fn")
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    buffer_factor = 2
    code = BINARY_FN_KERNEL_TEMPLATE.format(
        name=name,
        op=op,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        buffer_factor=buffer_factor,
        slice_syntax=_get_slice_syntax(tile_rows, tile_cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{name}_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = getattr(module, f"{name}_kernel")
    _kernel_cache[cache_key] = kernel
    return kernel


def make_unary_kernel(name: str, op: str, tile_rows: int, tile_cols: int) -> Callable:
    """Generate a unary kernel for the given shape."""
    cache_key = (name, op, tile_rows, tile_cols, "unary")
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    buffer_factor = 2
    code = UNARY_KERNEL_TEMPLATE.format(
        name=name,
        op=op,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        buffer_factor=buffer_factor,
        slice_syntax=_get_slice_syntax(tile_rows, tile_cols),
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{name}_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    _temp_files.append(temp_path)
    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = getattr(module, f"{name}_kernel")
    _kernel_cache[cache_key] = kernel
    return kernel


# =============================================================================
# Op Registry - maps op names to (kernel_factory, torch_fn)
# =============================================================================

BINARY_OPS = {
    "add": ("+", torch.add),
    "sub": ("-", torch.sub),
    "mul": ("*", torch.mul),
}

BINARY_FN_OPS = {
    "max": ("max", torch.maximum),
}

UNARY_OPS = {
    "exp": ("exp", torch.exp),
    "log": ("log", torch.log),
    "sqrt": ("sqrt", torch.sqrt),
    "rsqrt": ("rsqrt", torch.rsqrt),
    "tanh": ("tanh", torch.tanh),
    "abs": ("abs", torch.abs),
    "neg": ("neg", torch.neg),
    "relu": ("relu", torch.relu),
    "sigmoid": ("sigmoid", torch.sigmoid),
}


# =============================================================================
# Parametrized Tests - ops x shapes cross-product
# =============================================================================


@pytest.mark.parametrize("tile_shape", TILE_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}tiles")
@pytest.mark.parametrize("op_name", BINARY_OPS.keys())
def test_binary_op(device, op_name, tile_shape):
    """Test binary elementwise operation across different shapes."""
    tile_rows, tile_cols = tile_shape
    tensor_shape = tiles_to_tensor_shape(tile_rows, tile_cols)

    op_str, torch_fn = BINARY_OPS[op_name]
    kernel = make_binary_kernel(op_name, op_str, tile_rows, tile_cols)

    lhs_torch = torch.rand(tensor_shape, dtype=torch.bfloat16)
    rhs_torch = torch.rand(tensor_shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(tensor_shape, dtype=torch.bfloat16)
    expected = torch_fn(lhs_torch, rhs_torch)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tile_shape", TILE_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}tiles")
@pytest.mark.parametrize("op_name", BINARY_FN_OPS.keys())
def test_binary_fn_op(device, op_name, tile_shape):
    """Test binary elementwise operation (function call) across different shapes."""
    tile_rows, tile_cols = tile_shape
    tensor_shape = tiles_to_tensor_shape(tile_rows, tile_cols)

    op_str, torch_fn = BINARY_FN_OPS[op_name]
    kernel = make_binary_fn_kernel(op_name, op_str, tile_rows, tile_cols)

    lhs_torch = torch.rand(tensor_shape, dtype=torch.bfloat16)
    rhs_torch = torch.rand(tensor_shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(tensor_shape, dtype=torch.bfloat16)
    expected = torch_fn(lhs_torch, rhs_torch)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("tile_shape", TILE_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}tiles")
@pytest.mark.parametrize("op_name", UNARY_OPS.keys())
def test_unary_op(device, op_name, tile_shape):
    """Test unary elementwise operation across different shapes."""
    tile_rows, tile_cols = tile_shape
    tensor_shape = tiles_to_tensor_shape(tile_rows, tile_cols)

    op_str, torch_fn = UNARY_OPS[op_name]
    kernel = make_unary_kernel(op_name, op_str, tile_rows, tile_cols)

    # Use values appropriate for all ops (positive for log/sqrt, bounded for exp)
    inp_torch = torch.full(tensor_shape, 0.5, dtype=torch.bfloat16)
    out_torch = torch.zeros(tensor_shape, dtype=torch.bfloat16)
    expected = torch_fn(inp_torch)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
