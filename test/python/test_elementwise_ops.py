# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Auto-generated tests for TTL elementwise operations.

Tests all binary and unary elementwise ops against PyTorch equivalents,
with both L1 and DRAM memory configurations.
"""

# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

# NOT YET RUNNING this is just a placeholder for how this could look.

import pytest
import torch
import sys
from pathlib import Path

# Add examples to path for utils
# TODO: fix this
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
from utils import assert_allclose

from ttlang.ttl_api import (
    pykernel_gen,
    Program,
    CircularBuffer,
    TensorAccessor,
    compute,
    datamovement,
)
from ttlang.operators import copy
import ttlang

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False

# Skip all tests if ttnn not available
pytestmark = pytest.mark.skipif(not TTNN_AVAILABLE, reason="TTNN not available")


# =============================================================================
# Op Registry: Maps op names to (pytorch_fn, config)
# =============================================================================

BINARY_OPS = {
    "add": (torch.add, {"input_range": (-5, 5)}),
    "sub": (torch.sub, {"input_range": (-5, 5)}),
    "mul": (torch.mul, {"input_range": (-2, 2)}),
    "max": (torch.maximum, {"input_range": (-5, 5)}),
}

UNARY_OPS = {
    "exp": (torch.exp, {"input_range": (-2, 2)}),  # Avoid overflow
    "log": (torch.log, {"input_range": (0.1, 10)}),  # Positive only
    "sqrt": (torch.sqrt, {"input_range": (0.1, 10)}),  # Positive only
    "rsqrt": (torch.rsqrt, {"input_range": (0.1, 10)}),  # Positive only
    "tanh": (torch.tanh, {"input_range": (-3, 3)}),
    "abs": (torch.abs, {"input_range": (-5, 5)}),
    "neg": (torch.neg, {"input_range": (-5, 5)}),
    "relu": (torch.relu, {"input_range": (-5, 5)}),
    "sigmoid": (torch.sigmoid, {"input_range": (-5, 5)}),
}

MEMORY_CONFIGS = ["L1", "DRAM"]


# =============================================================================
# Kernel Factories - Generate kernels at module level
# =============================================================================

# TODO: doubt this will work in reality so we'll need to find another way to auto generate these.


def make_binary_kernel(op_name: str):
    """Factory to create a binary elementwise kernel for the given op."""
    op_fn = getattr(ttlang, op_name)

    @pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
    def binary_kernel(lhs, rhs, out):
        lhs_accessor = TensorAccessor(lhs)
        rhs_accessor = TensorAccessor(rhs)
        out_accessor = TensorAccessor(out)

        @compute()
        def compute_fn(
            lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
        ):
            l = lhs_cb.wait()
            r = rhs_cb.wait()
            o = out_cb.reserve()
            result = op_fn(l, r)
            o.store(result)
            lhs_cb.pop()
            rhs_cb.pop()
            out_cb.push()

        @datamovement()
        def dm_read(
            lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
        ):
            tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
            tx_lhs.wait()
            tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
            tx_rhs.wait()

        @datamovement()
        def dm_write(
            lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
        ):
            tx = copy(out_cb, out_accessor[0, 0])
            tx.wait()

        return Program(compute_fn, dm_read, dm_write)(lhs, rhs, out)

    return binary_kernel


def make_unary_kernel(op_name: str):
    """Factory to create a unary elementwise kernel for the given op."""
    op_fn = getattr(ttlang, op_name)

    @pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
    def unary_kernel(inp, out):
        inp_accessor = TensorAccessor(inp)
        out_accessor = TensorAccessor(out)

        @compute()
        def compute_fn(inp_cb: CircularBuffer, out_cb: CircularBuffer):
            x = inp_cb.wait()
            o = out_cb.reserve()
            result = op_fn(x)
            o.store(result)
            inp_cb.pop()
            out_cb.push()

        @datamovement()
        def dm_read(inp_cb: CircularBuffer, out_cb: CircularBuffer):
            tx = copy(inp_accessor[0, 0], inp_cb)
            tx.wait()

        @datamovement()
        def dm_write(inp_cb: CircularBuffer, out_cb: CircularBuffer):
            tx = copy(out_cb, out_accessor[0, 0])
            tx.wait()

        return Program(compute_fn, dm_read, dm_write)(inp, out)

    return unary_kernel


# Pre-generate all kernels at module level
BINARY_KERNELS = {name: make_binary_kernel(name) for name in BINARY_OPS}
UNARY_KERNELS = {name: make_unary_kernel(name) for name in UNARY_OPS}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def device():
    """Open device once per module."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


# =============================================================================
# Helper Functions
# =============================================================================


def create_tensor(torch_tensor, device, memory_config: str):
    """Create a ttnn tensor with the specified memory config."""
    mem_cfg = (
        ttnn.L1_MEMORY_CONFIG if memory_config == "L1" else ttnn.DRAM_MEMORY_CONFIG
    )

    tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Always start in DRAM
    )

    # Move to L1 if requested
    if memory_config == "L1":
        tensor = ttnn.to_memory_config(tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

    return tensor


def generate_input(input_range: tuple, shape=(32, 32), dtype=torch.bfloat16):
    """Generate random input tensor within the specified range."""
    low, high = input_range
    return torch.empty(shape, dtype=dtype).uniform_(low, high)


# =============================================================================
# Binary Op Tests
# =============================================================================


@pytest.mark.parametrize("op_name", BINARY_OPS.keys())
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_binary_op(device, op_name: str, memory_config: str):
    """Test binary elementwise operation against PyTorch."""
    torch_fn, config = BINARY_OPS[op_name]
    kernel = BINARY_KERNELS[op_name]
    input_range = config.get("input_range", (-5, 5))

    # Generate inputs
    lhs_torch = generate_input(input_range)
    rhs_torch = generate_input(input_range)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result
    expected = torch_fn(lhs_torch, rhs_torch)

    # Create device tensors
    lhs = create_tensor(lhs_torch, device, memory_config)
    rhs = create_tensor(rhs_torch, device, memory_config)
    out = create_tensor(out_torch, device, memory_config)

    # Run kernel
    kernel(lhs, rhs, out)

    # Get result
    result = ttnn.to_torch(out)

    # Compare with tolerance appropriate for bfloat16
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Unary Op Tests
# =============================================================================


@pytest.mark.parametrize("op_name", UNARY_OPS.keys())
@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_unary_op(device, op_name: str, memory_config: str):
    """Test unary elementwise operation against PyTorch."""
    torch_fn, config = UNARY_OPS[op_name]
    kernel = UNARY_KERNELS[op_name]
    input_range = config.get("input_range", (-5, 5))

    # Generate input
    inp_torch = generate_input(input_range)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result
    expected = torch_fn(inp_torch)

    # Create device tensors
    inp = create_tensor(inp_torch, device, memory_config)
    out = create_tensor(out_torch, device, memory_config)

    # Run kernel
    kernel(inp, out)

    # Get result
    result = ttnn.to_torch(out)

    # Compare with tolerance appropriate for bfloat16
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Main - for running outside pytest
# =============================================================================

if __name__ == "__main__":
    if not TTNN_AVAILABLE:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    print("=== Elementwise Ops Test Suite ===\n")

    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
