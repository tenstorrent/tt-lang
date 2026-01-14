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

    return ttl.Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)
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

    return ttl.Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)
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

    return ttl.Program(compute_fn, dm_read, dm_write)(inp, out)
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
# Power Kernel Template (unary with scalar exponent)
# =============================================================================

POWER_KERNEL_TEMPLATE = '''
from ttlang import ttl

@ttl.kernel(grid=(1, 1))
def power_kernel(inp, out):
    """Power kernel with scalar exponent (exponent baked into kernel)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = ttl.power(x, {exponent})
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

    return ttl.Program(compute_fn, dm_read, dm_write)(inp, out)
'''


def make_power_kernel(exponent: int):
    """Generate a power kernel with given exponent."""
    code = POWER_KERNEL_TEMPLATE.format(exponent=exponent)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix=f"kernel_power_{exponent}_"
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location(f"power_{exponent}_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "power_kernel")


# =============================================================================
# Where Kernel Template (ternary conditional)
# =============================================================================

WHERE_KERNEL_TEMPLATE = '''
from ttlang import ttl

@ttl.kernel(grid=(1, 1))
def where_kernel(cond, true_val, false_val, out):
    """Where/select kernel."""
    cond_cb = ttl.make_circular_buffer_like(cond, shape=(1, 1), buffer_factor=2)
    true_cb = ttl.make_circular_buffer_like(true_val, shape=(1, 1), buffer_factor=2)
    false_cb = ttl.make_circular_buffer_like(false_val, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        c = cond_cb.wait()
        t = true_cb.wait()
        f = false_cb.wait()
        o = out_cb.reserve()
        result = ttl.where(c, t, f)
        o.store(result)
        cond_cb.pop()
        true_cb.pop()
        false_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        cond_blk = cond_cb.reserve()
        tx_cond = ttl.copy(cond[0, 0], cond_blk)
        tx_cond.wait()
        cond_cb.push()

        true_blk = true_cb.reserve()
        tx_true = ttl.copy(true_val[0, 0], true_blk)
        tx_true.wait()
        true_cb.push()

        false_blk = false_cb.reserve()
        tx_false = ttl.copy(false_val[0, 0], false_blk)
        tx_false.wait()
        false_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(cond, true_val, false_val, out)
'''


def make_where_kernel():
    """Generate a where/select kernel."""
    code = WHERE_KERNEL_TEMPLATE

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_where_"
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("where_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "where_kernel")


# =============================================================================
# Op Registry - maps op names to (kernel, torch_fn)
# =============================================================================

BINARY_OPS = {
    "add": (make_binary_kernel("add", "+"), torch.add),
    "sub": (make_binary_kernel("sub", "-"), torch.sub),
    "mul": (make_binary_kernel("mul", "*"), torch.mul),
    "max": (make_binary_fn_kernel("max", "max"), torch.maximum),
    "min": (make_binary_fn_kernel("min", "min"), torch.minimum),
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
    "floor": (make_unary_kernel("floor", "floor"), torch.floor),
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


# =============================================================================
# Power Test
# =============================================================================


@pytest.mark.parametrize("exponent", [2, 3])
def test_power_op(device, exponent):
    """Test power operation with scalar exponent."""
    kernel = make_power_kernel(exponent)

    # Generate inputs - use values appropriate for power
    inp_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result
    expected = torch.pow(inp_torch, exponent)

    # Create device tensors
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

    # Move to L1
    inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel (exponent is baked into the kernel)
    kernel(inp, out)

    # Get result
    result = ttnn.to_torch(out)

    # Compare with tolerance appropriate for bfloat16
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Where Test
# =============================================================================


@pytest.mark.skip(reason="where_tile uses sfpsetcc instruction not implemented in simulator - test hangs")
def test_where_op(device):
    """Test where/select operation."""
    kernel = make_where_kernel()

    # Generate inputs
    # Condition: half 1s, half 0s
    cond_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    cond_torch[:16, :] = 1.0  # Top half is true
    true_torch = torch.full((32, 32), 5.0, dtype=torch.bfloat16)
    false_torch = torch.full((32, 32), 10.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result
    expected = torch.where(cond_torch.bool(), true_torch, false_torch)

    # Create device tensors
    cond = ttnn.from_torch(
        cond_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    true_val = ttnn.from_torch(
        true_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    false_val = ttnn.from_torch(
        false_torch,
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

    # Move to L1
    cond = ttnn.to_memory_config(cond, memory_config=ttnn.L1_MEMORY_CONFIG)
    true_val = ttnn.to_memory_config(true_val, memory_config=ttnn.L1_MEMORY_CONFIG)
    false_val = ttnn.to_memory_config(false_val, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    kernel(cond, true_val, false_val, out)

    # Get result
    result = ttnn.to_torch(out)

    # Compare with tolerance appropriate for bfloat16
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Main - for running outside pytest
# =============================================================================

if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
