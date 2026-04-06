# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTL elementwise operations with ND tensor shapes.

Tests binary and unary eltwise ops with 2D, 3D, 4D, and 5D tensor shapes to
verify the compiler handles arbitrary-rank tensors correctly. Each test compiles
a kernel with the appropriate shape, CB rank, and indexing.

Parameterized over:
- Operations: binary (add, sub, mul, max) and unary (exp, log, sqrt, etc.)
- Shapes: Various 2D through 5D configurations
"""

# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import importlib.util
import tempfile
from typing import Callable, Tuple

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttlang_test_utils import assert_allclose, to_dram

TILE_SIZE = 32


def tiles(n: int) -> int:
    """Convert tile count to element count."""
    return n * TILE_SIZE


# Shapes as tuples of element dimensions. The last 2 dims are tiled (must be
# multiples of TILE_SIZE). Earlier dims are batch dims.
SHAPES = [
    # 2D shapes
    (32, 32),
    (64, 32),
    (64, 64),
    # 3D shapes
    (1, 32, 32),
    (2, 32, 32),
    (3, 64, 32),
    (2, 64, 64),
    # 4D shapes
    (1, 1, 32, 32),
    (2, 2, 32, 32),
    (1, 3, 64, 32),
    (2, 2, 64, 64),
    # 5D shapes
    (1, 1, 1, 32, 32),
    (2, 1, 1, 32, 32),
    (1, 2, 3, 32, 32),
]

# Only test ND (3D+) shapes to keep the sweep focused on what's new.
# 2D shapes are already covered by test_elementwise_shapes.py.
ND_SHAPES = [s for s in SHAPES if len(s) > 2]

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
# Code generation helpers
# =============================================================================


def _shape_to_tile_grid(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert element shape to tile grid shape."""
    batch = shape[:-2]
    rows = shape[-2] // TILE_SIZE
    cols = shape[-1] // TILE_SIZE
    return batch + (rows, cols)


def _make_loop_nest(grid, rank, body_lines, base_indent="        "):
    """Generate nested for loops with the given body lines."""
    lines = []
    for i, dim in enumerate(grid):
        indent = base_indent + "    " * i
        lines.append(f"{indent}for d{i} in range({dim}):")
    body_indent = base_indent + "    " * rank
    for line in body_lines:
        lines.append(f"{body_indent}{line}")
    return "\n".join(lines)


def _idx_str(rank):
    return ", ".join(f"d{i}" for i in range(rank))


def _cb_shape_str(rank):
    return ", ".join(["1"] * rank)


# =============================================================================
# Kernel template factories
# =============================================================================


def _make_binary_kernel_code(shape, op_str):
    grid = _shape_to_tile_grid(shape)
    rank = len(grid)
    cb = _cb_shape_str(rank)
    idx = _idx_str(rank)

    compute_body = [
        "l = lhs_dfb.wait()",
        "r = rhs_dfb.wait()",
        "o = out_dfb.reserve()",
        f"result = l {op_str} r",
        "o.store(result)",
        "l.pop()",
        "r.pop()",
        "o.push()",
    ]
    dm_read_body = [
        "lhs_blk = lhs_dfb.reserve()",
        f"tx_lhs = ttl.copy(lhs[{idx}], lhs_blk)",
        "tx_lhs.wait()",
        "lhs_blk.push()",
        "",
        "rhs_blk = rhs_dfb.reserve()",
        f"tx_rhs = ttl.copy(rhs[{idx}], rhs_blk)",
        "tx_rhs.wait()",
        "rhs_blk.push()",
    ]
    dm_write_body = [
        "out_blk = out_dfb.wait()",
        f"tx = ttl.copy(out_blk, out[{idx}])",
        "tx.wait()",
        "out_blk.pop()",
    ]

    return f"""\
import ttl

@ttl.operation(grid=(1, 1))
def nd_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=({cb}), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=({cb}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({cb}), block_count=2)

    @ttl.compute()
    def compute_fn():
{_make_loop_nest(grid, rank, compute_body)}

    @ttl.datamovement()
    def dm_read():
{_make_loop_nest(grid, rank, dm_read_body)}

    @ttl.datamovement()
    def dm_write():
{_make_loop_nest(grid, rank, dm_write_body)}
"""


def _make_binary_fn_kernel_code(shape, fn_name):
    grid = _shape_to_tile_grid(shape)
    rank = len(grid)
    cb = _cb_shape_str(rank)
    idx = _idx_str(rank)

    compute_body = [
        "l = lhs_dfb.wait()",
        "r = rhs_dfb.wait()",
        "o = out_dfb.reserve()",
        f"result = ttl.math.{fn_name}(l, r)",
        "o.store(result)",
        "l.pop()",
        "r.pop()",
        "o.push()",
    ]
    dm_read_body = [
        "lhs_blk = lhs_dfb.reserve()",
        f"tx_lhs = ttl.copy(lhs[{idx}], lhs_blk)",
        "tx_lhs.wait()",
        "lhs_blk.push()",
        "",
        "rhs_blk = rhs_dfb.reserve()",
        f"tx_rhs = ttl.copy(rhs[{idx}], rhs_blk)",
        "tx_rhs.wait()",
        "rhs_blk.push()",
    ]
    dm_write_body = [
        "out_blk = out_dfb.wait()",
        f"tx = ttl.copy(out_blk, out[{idx}])",
        "tx.wait()",
        "out_blk.pop()",
    ]

    return f"""\
import ttl

@ttl.operation(grid=(1, 1))
def nd_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=({cb}), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=({cb}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({cb}), block_count=2)

    @ttl.compute()
    def compute_fn():
{_make_loop_nest(grid, rank, compute_body)}

    @ttl.datamovement()
    def dm_read():
{_make_loop_nest(grid, rank, dm_read_body)}

    @ttl.datamovement()
    def dm_write():
{_make_loop_nest(grid, rank, dm_write_body)}
"""


def _make_unary_kernel_code(shape, fn_name):
    grid = _shape_to_tile_grid(shape)
    rank = len(grid)
    cb = _cb_shape_str(rank)
    idx = _idx_str(rank)

    compute_body = [
        "x = inp_dfb.wait()",
        "o = out_dfb.reserve()",
        f"result = ttl.math.{fn_name}(x)",
        "o.store(result)",
        "x.pop()",
        "o.push()",
    ]
    dm_read_body = [
        "inp_blk = inp_dfb.reserve()",
        f"tx_inp = ttl.copy(inp[{idx}], inp_blk)",
        "tx_inp.wait()",
        "inp_blk.push()",
    ]
    dm_write_body = [
        "out_blk = out_dfb.wait()",
        f"tx = ttl.copy(out_blk, out[{idx}])",
        "tx.wait()",
        "out_blk.pop()",
    ]

    return f"""\
import ttl

@ttl.operation(grid=(1, 1))
def nd_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=({cb}), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=({cb}), block_count=2)

    @ttl.compute()
    def compute_fn():
{_make_loop_nest(grid, rank, compute_body)}

    @ttl.datamovement()
    def dm_read():
{_make_loop_nest(grid, rank, dm_read_body)}

    @ttl.datamovement()
    def dm_write():
{_make_loop_nest(grid, rank, dm_write_body)}
"""


# =============================================================================
# Kernel cache and loader
# =============================================================================

_kernel_cache: dict[tuple, Callable] = {}


def _load_kernel(cache_key, code, prefix):
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        prefix=prefix,
    ) as f:
        f.write(code)
        temp_path = f.name

    temp_kernel_files.append(temp_path)
    spec = importlib.util.spec_from_file_location("nd_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernel = module.nd_kernel
    _kernel_cache[cache_key] = kernel
    return kernel


# =============================================================================
# Tests
# =============================================================================


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


@pytest.mark.parametrize("shape", ND_SHAPES, ids=[_shape_id(s) for s in ND_SHAPES])
@pytest.mark.parametrize("op_name", BINARY_OPS.keys())
def test_nd_binary_op(device, op_name, shape):
    """Test binary elementwise op with ND tensor shape."""
    op_str, torch_fn = BINARY_OPS[op_name]
    code = _make_binary_kernel_code(shape, op_str)
    kernel = _load_kernel(("binary", op_name, shape), code, f"kernel_nd_{op_name}_")

    lhs_torch = torch.rand(shape, dtype=torch.bfloat16)
    rhs_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(shape, dtype=torch.bfloat16)
    expected = torch_fn(lhs_torch, rhs_torch)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", ND_SHAPES, ids=[_shape_id(s) for s in ND_SHAPES])
@pytest.mark.parametrize("op_name", BINARY_FN_OPS.keys())
def test_nd_binary_fn_op(device, op_name, shape):
    """Test binary function-call elementwise op with ND tensor shape."""
    fn_name, torch_fn = BINARY_FN_OPS[op_name]
    code = _make_binary_fn_kernel_code(shape, fn_name)
    kernel = _load_kernel(("binary_fn", op_name, shape), code, f"kernel_nd_{op_name}_")

    lhs_torch = torch.rand(shape, dtype=torch.bfloat16)
    rhs_torch = torch.rand(shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(shape, dtype=torch.bfloat16)
    expected = torch_fn(lhs_torch, rhs_torch)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", ND_SHAPES, ids=[_shape_id(s) for s in ND_SHAPES])
@pytest.mark.parametrize("op_name", UNARY_OPS.keys())
def test_nd_unary_op(device, op_name, shape):
    """Test unary elementwise op with ND tensor shape."""
    fn_name, torch_fn = UNARY_OPS[op_name]
    code = _make_unary_kernel_code(shape, fn_name)
    kernel = _load_kernel(("unary", op_name, shape), code, f"kernel_nd_{op_name}_")

    # Use values safe for all ops (positive for log/sqrt, bounded for exp)
    inp_torch = torch.full(shape, 0.5, dtype=torch.bfloat16)
    out_torch = torch.zeros(shape, dtype=torch.bfloat16)
    expected = torch_fn(inp_torch)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
