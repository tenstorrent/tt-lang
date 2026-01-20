# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTL elementwise operations.

Tests elementwise ops against PyTorch equivalents with L1 memory configuration.
Kernels are generated from a template, written to temp files, and imported.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import importlib.util
import tempfile

import pytest
import torch
import ttnn
from test_helpers import assert_allclose, to_l1

pytestmark = pytest.mark.requires_ttnn


# =============================================================================
# Kernel Template - generates kernels via temp file + import
# =============================================================================

BINARY_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

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
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

'''

BINARY_FN_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel (function call)."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

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
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

'''

UNARY_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def {name}_kernel(inp, out):
    """Unary {name} kernel."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

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
        tx_inp = ttl.copy(inp[0, 0], inp_blk)
        tx_inp.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

'''


def make_binary_kernel(name: str, op: str):
    """Generate a binary kernel by writing to temp file and importing."""
    code = BINARY_KERNEL_TEMPLATE.format(name=name, op=op)

    # Write to temp file (delete=False so we can import it)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"kernel_{name}_"
    ) as f:
        f.write(code)
        temp_path = f.name

    # Import the module
    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, f"{name}_kernel")


def make_binary_fn_kernel(name: str, op: str):
    """Generate a binary kernel using function call syntax (e.g., ttl.math.max(l, r))."""
    code = BINARY_FN_KERNEL_TEMPLATE.format(name=name, op=op)

    # Write to temp file (delete=False so we can import it)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"kernel_{name}_"
    ) as f:
        f.write(code)
        temp_path = f.name

    # Import the module
    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, f"{name}_kernel")


def make_unary_kernel(name: str, op: str):
    """Generate a unary kernel by writing to temp file and importing."""
    code = UNARY_KERNEL_TEMPLATE.format(name=name, op=op)

    # Write to temp file (delete=False so we can import it)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"kernel_{name}_"
    ) as f:
        f.write(code)
        temp_path = f.name

    # Import the module
    spec = importlib.util.spec_from_file_location(f"{name}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, f"{name}_kernel")


# =============================================================================
# Op Registry - maps op names to (kernel, torch_fn)
# =============================================================================

BINARY_OPS = {
    "add": (make_binary_kernel("add", "+"), torch.add),
    "sub": (make_binary_kernel("sub", "-"), torch.sub),
    "mul": (make_binary_kernel("mul", "*"), torch.mul),
    "max": (make_binary_fn_kernel("max", "max"), torch.maximum),
}

UNARY_OPS = {
    "exp": (make_unary_kernel("exp", "exp"), torch.exp),
    "log": (make_unary_kernel("log", "log"), torch.log),
    "sqrt": (make_unary_kernel("sqrt", "sqrt"), torch.sqrt),
    "rsqrt": (make_unary_kernel("rsqrt", "rsqrt"), torch.rsqrt),
    "tanh": (make_unary_kernel("tanh", "tanh"), torch.tanh),
    "abs": (make_unary_kernel("abs", "abs"), torch.abs),
    "neg": (make_unary_kernel("neg", "neg"), torch.neg),
    "relu": (make_unary_kernel("relu", "relu"), torch.relu),
    "sigmoid": (make_unary_kernel("sigmoid", "sigmoid"), torch.sigmoid),
}


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize("op_name", BINARY_OPS.keys())
def test_binary_op(device, op_name):
    """Test binary elementwise operation with L1 memory."""
    kernel, torch_fn = BINARY_OPS[op_name]

    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch_fn(lhs_torch, rhs_torch)

    lhs = to_l1(lhs_torch, device)
    rhs = to_l1(rhs_torch, device)
    out = to_l1(out_torch, device)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("op_name", UNARY_OPS.keys())
def test_unary_op(device, op_name):
    """Test unary elementwise operation with L1 memory."""
    kernel, torch_fn = UNARY_OPS[op_name]

    # Use values appropriate for all ops (positive for log/sqrt, bounded for exp)
    inp_torch = torch.full((32, 32), 0.5, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch_fn(inp_torch)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
