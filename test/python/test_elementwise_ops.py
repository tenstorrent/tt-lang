# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from conftest import temp_kernel_files
from ttlang_test_utils import assert_allclose, to_l1, to_l1_sharded
from utils.correctness import assert_with_ulp

# =============================================================================
# Kernel Template - generates kernels via temp file + import
# =============================================================================

BINARY_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        result = l {op} r
        o.store(result)
        l.pop()
        r.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], lhs_blk).wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], rhs_blk).wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        ttl.copy(out_blk, out[0, 0]).wait()
        out_blk.pop()

'''

BINARY_FN_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def {name}_kernel(lhs, rhs, out):
    """Binary {name} kernel (function call)."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        result = ttl.math.{op}(l, r)
        o.store(result)
        l.pop()
        r.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], lhs_blk).wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], rhs_blk).wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        ttl.copy(out_blk, out[0, 0]).wait()
        out_blk.pop()

'''

UNARY_KERNEL_TEMPLATE = '''
import ttl

@ttl.operation(grid=(1, 1))
def {name}_kernel(inp, out):
    """Unary {name} kernel."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        x = inp_dfb.wait()
        o = out_dfb.reserve()
        result = ttl.math.{op}(x)
        o.store(result)
        x.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        ttl.copy(inp[0, 0], inp_blk).wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        ttl.copy(out_blk, out[0, 0]).wait()
        out_blk.pop()

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
    temp_kernel_files.append(temp_path)

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
    temp_kernel_files.append(temp_path)

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
    temp_kernel_files.append(temp_path)

    return getattr(module, f"{name}_kernel")


# =============================================================================
# Op Registry - maps op names to (kernel, torch_fn)
# =============================================================================

BINARY_OPS = {
    "add": (make_binary_kernel("add", "+"), torch.add),
    "sub": (make_binary_kernel("sub", "-"), torch.sub),
    "mul": (make_binary_kernel("mul", "*"), torch.mul),
    "div": (make_binary_kernel("div", "/"), torch.div),
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
    "recip": (make_unary_kernel("recip", "recip"), torch.reciprocal),
    "sin": (make_unary_kernel("sin", "sin"), torch.sin),
    "cos": (make_unary_kernel("cos", "cos"), torch.cos),
    "tan": (make_unary_kernel("tan", "tan"), torch.tan),
    "asin": (make_unary_kernel("asin", "asin"), torch.asin),
    "acos": (make_unary_kernel("acos", "acos"), torch.acos),
    "atan": (make_unary_kernel("atan", "atan"), torch.atan),
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
# Sharded L1 Tests - same ops with height-sharded memory layout
# =============================================================================


@pytest.mark.parametrize("shard_layout", ["height", "width", "block"])
@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
def test_binary_op_sharded(device, op_name, shard_layout):
    """Test binary elementwise operation with sharded L1 memory."""
    kernel, torch_fn = BINARY_OPS[op_name]

    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch_fn(lhs_torch, rhs_torch)

    lhs = to_l1_sharded(lhs_torch, device, layout=shard_layout)
    rhs = to_l1_sharded(rhs_torch, device, layout=shard_layout)
    out = to_l1_sharded(out_torch, device, layout=shard_layout)

    kernel(lhs, rhs, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shard_layout", ["height", "width", "block"])
@pytest.mark.parametrize("op_name", ["exp", "neg", "relu"])
def test_unary_op_sharded(device, op_name, shard_layout):
    """Test unary elementwise operation with sharded L1 memory."""
    kernel, torch_fn = UNARY_OPS[op_name]

    inp_torch = torch.full((32, 32), 0.5, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch_fn(inp_torch)

    inp = to_l1_sharded(inp_torch, device, layout=shard_layout)
    out = to_l1_sharded(out_torch, device, layout=shard_layout)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Inverse trig tests - bf16 bit patterns with ULP validation matching tt-metal
# =============================================================================


def _flush_subnormals(tensor):
    """Flush subnormal values to zero in-place (hardware behavior)."""
    tensor[torch.abs(tensor) < 2.0 ** (-126)] = 0.0
    return tensor


def _bf16_bitpatterns_in_range(lo, hi, dtype):
    """Generate all bf16 bit patterns in [lo, hi], cast to dtype.

    Matches tt-metal's test methodology (generate_all_bfloat16_bitpatterns)
    but filtered to a domain range and sized for a single 32x32 tile.
    Hardware flushes subnormals to zero, so we do the same.
    """
    all_bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    all_bf16 = all_bits.view(torch.bfloat16).to(torch.float32)
    # Filter to domain and remove non-finite values.
    mask = (all_bf16 >= lo) & (all_bf16 <= hi) & torch.isfinite(all_bf16)
    values = all_bf16[mask]
    _flush_subnormals(values)
    # Sample down to 32*32 = 1024 if needed, preserving endpoints.
    num_elements = 32 * 32
    if len(values) > num_elements:
        indices = torch.linspace(0, len(values) - 1, steps=num_elements).long()
        values = values[indices]
    elif len(values) < num_elements:
        # Pad with zeros if too few values.
        values = torch.nn.functional.pad(values, (0, num_elements - len(values)))
    return values.to(dtype).reshape(32, 32)


# Thresholds aligned with tt-metal tests:
#   bf16: asin ULP 3, acos ULP 3, atan ULP 3 (test_unary.py)
#   f32:  asin ULP 100, acos ULP 100, atan ULP 3 (test_unary_fp32.py)
INVERSE_TRIG_OPS = {
    "asin": {
        "torch_fn": torch.asin,
        # Exclude exact boundaries where derivative is infinite.
        "input_range": (-0.9961, 0.9961),
        # bf16 ULP 3: aligned with tt-metal run_unary_inverse_trig_bf16_test.
        # f32 ULP 2^15: true f32 linspace hits SFPU polynomial precision limits
        # (measured max ~14748); tt-metal's ULP 100 only applies to bf16 bit patterns.
        "ulp": {torch.bfloat16: 3, torch.float32: 2**15},
    },
    "acos": {
        "torch_fn": torch.acos,
        "input_range": (-0.9961, 0.9961),
        # Same precision characteristics as asin.
        "ulp": {torch.bfloat16: 3, torch.float32: 2**15},
    },
    "atan": {
        "torch_fn": torch.atan,
        "input_range": (-10.0, 10.0),
        "ulp": {torch.bfloat16: 3, torch.float32: 2**14},
    },
}


@pytest.mark.parametrize("op_name", INVERSE_TRIG_OPS.keys())
def test_inverse_trig_bf16(device, op_name):
    """Test inverse trig (bf16) with bf16 bit pattern inputs and ULP validation.

    Matches tt-metal's test methodology: bf16 bit patterns with subnormal
    flushing, filtered to valid domain. ULP thresholds aligned with
    tt-metal's test_unary.py (run_unary_inverse_trig_bf16_test).
    """
    spec = INVERSE_TRIG_OPS[op_name]
    kernel, _ = UNARY_OPS[op_name]
    lo, hi = spec["input_range"]
    dtype = torch.bfloat16

    inp_torch = _bf16_bitpatterns_in_range(lo, hi, dtype)
    out_torch = torch.zeros((32, 32), dtype=dtype)
    expected = spec["torch_fn"](inp_torch)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_with_ulp(expected, result, ulp_threshold=spec["ulp"][dtype])


@pytest.mark.parametrize("op_name", INVERSE_TRIG_OPS.keys())
def test_inverse_trig_f32(device, op_name):
    """Test inverse trig (f32) with dense linspace inputs and ULP validation.

    Uses linspace to generate evenly-spaced f32 values across the valid
    domain, avoiding the zero-density problem of bf16 bit pattern
    subsampling. ULP at zero is meaningless (ULP(0)=1.4e-45), so the
    range excludes zero. Thresholds calibrated from measured hardware
    precision (higher than tt-metal's bf16-pattern thresholds).
    """
    spec = INVERSE_TRIG_OPS[op_name]
    kernel, _ = UNARY_OPS[op_name]
    lo, hi = spec["input_range"]
    dtype = torch.float32

    # Two linspace halves to avoid zero (ULP is meaningless at zero).
    neg_half = torch.linspace(lo, -0.001, steps=512, dtype=dtype)
    pos_half = torch.linspace(0.001, hi, steps=512, dtype=dtype)
    inp_torch = torch.cat([neg_half, pos_half]).reshape(32, 32)
    out_torch = torch.zeros((32, 32), dtype=dtype)
    expected = spec["torch_fn"](inp_torch)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_with_ulp(expected, result, ulp_threshold=spec["ulp"][dtype])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
