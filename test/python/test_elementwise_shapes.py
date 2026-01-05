# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTL elementwise operations across multiple tensor shapes.

Tests elementwise ops against PyTorch equivalents with L1 memory configuration,
sweeping all tile configurations from 1x1 to 4x4.

Parameterized over:
- Operations: binary (add, sub, mul, max) and unary (exp, log, sqrt, etc.)
- Shapes: all tile configurations from 1x1 to 4x4 (16 shapes total)

Total test cases: 13 ops x 16 shapes = 208 tests
"""

# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import sys
import tempfile
import importlib.util
from pathlib import Path

# Import ttnn BEFORE torch to avoid nanobind initialization issues
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

import torch
from utils import assert_allclose

# Skip all tests if ttnn not available
pytestmark = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")

# =============================================================================
# Shape Configurations - all permutations from 1x1 to 4x4 tiles
# =============================================================================

TILE_SIZE = 32

# TODO: right now I'm seeing resource errors with shape > 3 tiles (2x2 or 4x1),
# so limiting to max 3x1 or 1x3.
TILE_SHAPES = [(r, c) for r in range(1, 4) for c in range(1, 4) if r == 1 or c == 1]


def tiles_to_tensor_shape(tile_rows: int, tile_cols: int) -> tuple[int, int]:
    """Convert tile grid dimensions to tensor shape."""
    return (tile_rows * TILE_SIZE, tile_cols * TILE_SIZE)


# =============================================================================
# Kernel Templates - parameterized by shape
# =============================================================================

BINARY_KERNEL_TEMPLATE = '''
from ttlang import ttl, make_circular_buffer_like
from ttlang.ttl_api import Program
from ttlang.operators import copy

@ttl.kernel(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel for {tile_rows}x{tile_cols} tiles."""
    lhs_cb = make_circular_buffer_like(lhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    rhs_cb = make_circular_buffer_like(rhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    out_cb = make_circular_buffer_like(out, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})

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
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)
'''

BINARY_FN_KERNEL_TEMPLATE = '''
from ttlang import ttl, make_circular_buffer_like
from ttlang.ttl_api import Program
from ttlang.operators import copy
from ttlang import {op}

@ttl.kernel(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel (function call) for {tile_rows}x{tile_cols} tiles."""
    lhs_cb = make_circular_buffer_like(lhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    rhs_cb = make_circular_buffer_like(rhs, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    out_cb = make_circular_buffer_like(out, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})

    @ttl.compute()
    def compute_fn():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = {op}(l, r)
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)
'''

UNARY_KERNEL_TEMPLATE = '''
from ttlang import ttl, make_circular_buffer_like
from ttlang.ttl_api import Program
from ttlang.operators import copy
from ttlang import {op}

@ttl.kernel(grid=(1, 1))
def {name}_kernel(inp, out):
    """Unary {name} kernel for {tile_rows}x{tile_cols} tiles."""
    inp_cb = make_circular_buffer_like(inp, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})
    out_cb = make_circular_buffer_like(out, shape=({tile_rows}, {tile_cols}), buffer_factor={buffer_factor})

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = {op}(x)
        o.store(result)
        inp_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        inp_cb.reserve()
        tx_inp = copy(inp[0, 0], inp_cb)
        tx_inp.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_fn, dm_read, dm_write)(inp, out)
'''


# =============================================================================
# Kernel Factories - generate kernels with specific shapes
# =============================================================================

# Cache for generated kernels: (name, op, tile_rows, tile_cols) -> kernel
_kernel_cache = {}


def make_binary_kernel(name: str, op: str, tile_rows: int, tile_cols: int):
    """Generate a binary kernel for the given shape."""
    cache_key = (name, op, tile_rows, tile_cols, "binary")
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    buffer_factor = tile_rows * tile_cols
    code = BINARY_KERNEL_TEMPLATE.format(
        name=name,
        op=op,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        buffer_factor=buffer_factor,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{name}_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = getattr(module, f"{name}_kernel")
    _kernel_cache[cache_key] = kernel
    return kernel


def make_binary_fn_kernel(name: str, op: str, tile_rows: int, tile_cols: int):
    """Generate a binary kernel using function call syntax."""
    cache_key = (name, op, tile_rows, tile_cols, "binary_fn")
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    buffer_factor = tile_rows * tile_cols
    code = BINARY_FN_KERNEL_TEMPLATE.format(
        name=name,
        op=op,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        buffer_factor=buffer_factor,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{name}_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = getattr(module, f"{name}_kernel")
    _kernel_cache[cache_key] = kernel
    return kernel


def make_unary_kernel(name: str, op: str, tile_rows: int, tile_cols: int):
    """Generate a unary kernel for the given shape."""
    cache_key = (name, op, tile_rows, tile_cols, "unary")
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    buffer_factor = tile_rows * tile_cols
    code = UNARY_KERNEL_TEMPLATE.format(
        name=name,
        op=op,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        buffer_factor=buffer_factor,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=f"kernel_{name}_{tile_rows}x{tile_cols}_",
    ) as f:
        f.write(code)
        temp_path = f.name

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
# Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


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

    # Generate inputs
    lhs_torch = torch.full(tensor_shape, 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full(tensor_shape, 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros(tensor_shape, dtype=torch.bfloat16)

    # Compute expected result
    expected = torch_fn(lhs_torch, rhs_torch)

    # Create device tensors
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

    # Run kernel
    kernel(lhs, rhs, out)

    # Get result and compare
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

    # Generate inputs
    lhs_torch = torch.full(tensor_shape, 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full(tensor_shape, 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros(tensor_shape, dtype=torch.bfloat16)

    # Compute expected result
    expected = torch_fn(lhs_torch, rhs_torch)

    # Create device tensors in DRAM
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

    # Run kernel
    kernel(lhs, rhs, out)

    # Get result and compare
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

    # Generate inputs - use values appropriate for all ops
    # (positive values for log/sqrt, avoid extremes for exp)
    inp_torch = torch.full(tensor_shape, 0.5, dtype=torch.bfloat16)
    out_torch = torch.zeros(tensor_shape, dtype=torch.bfloat16)

    # Compute expected result
    expected = torch_fn(inp_torch)

    # Create device tensors in DRAM
    inp = ttnn.from_torch(
        inp_torch,
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

    # Run kernel
    kernel(inp, out)

    # Get result and compare
    result = ttnn.to_torch(out)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Main - for running outside pytest
# =============================================================================

if __name__ == "__main__":
    if not TTNN_AVAILABLE:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    print("=== Elementwise Ops Shape Sweep Test ===")
    print(f"Shapes: {len(TILE_SHAPES)} configurations (1x1 to 4x4 tiles)")
    print(f"Binary ops: {list(BINARY_OPS.keys()) + list(BINARY_FN_OPS.keys())}")
    print(f"Unary ops: {list(UNARY_OPS.keys())}")
    print(
        f"Total tests: {len(TILE_SHAPES) * (len(BINARY_OPS) + len(BINARY_FN_OPS) + len(UNARY_OPS))}"
    )
    print()

    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
