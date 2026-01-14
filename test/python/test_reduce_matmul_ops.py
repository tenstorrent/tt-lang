# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for TTL reduce and matmul operations.

Tests matmul, reduce_sum, reduce_max, and transpose ops against PyTorch equivalents.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import importlib.util
import tempfile

import pytest
import torch
import ttnn
from test_helpers import assert_allclose

# Skip all tests if ttnn not available
pytestmark = pytest.mark.requires_ttnn


# =============================================================================
# Matmul Kernel Template
# =============================================================================

MATMUL_KERNEL_TEMPLATE = '''
from ttlang import ttl

@ttl.kernel(grid=(1, 1))
def matmul_kernel(a, b, out):
    """Single-tile matmul kernel: out = a @ b."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        a_tile = a_cb.wait()
        b_tile = b_cb.wait()
        o = out_cb.reserve()
        result = ttl.matmul(a_tile, b_tile, o)
        o.store(result)
        a_cb.pop()
        b_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(a, b, out)
'''


def make_matmul_kernel():
    """Generate matmul kernel by writing to temp file and importing."""
    code = MATMUL_KERNEL_TEMPLATE

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_matmul_"
    ) as f:
        f.write(code)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("matmul_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "matmul_kernel")


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
# Matmul Test
# =============================================================================


def test_matmul_op(device):
    """Test single-tile matmul operation."""
    kernel = make_matmul_kernel()

    # Single tile: 32x32 @ 32x32 = 32x32
    a_torch = torch.randn((32, 32), dtype=torch.bfloat16) * 0.1
    b_torch = torch.randn((32, 32), dtype=torch.bfloat16) * 0.1
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result
    expected = torch.matmul(a_torch.float(), b_torch.float()).to(torch.bfloat16)

    # Create device tensors
    a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        b_torch,
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
    a = ttnn.to_memory_config(a, memory_config=ttnn.L1_MEMORY_CONFIG)
    b = ttnn.to_memory_config(b, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    kernel(a, b, out)

    # Get result
    result = ttnn.to_torch(out)

    # Compare with relaxed tolerance for matmul
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Reduce Tests (placeholder - to be implemented)
# =============================================================================


@pytest.mark.skip(reason="reduce_sum not yet implemented")
def test_reduce_sum_op(device):
    """Test reduce_sum operation."""
    pass


@pytest.mark.skip(reason="reduce_max not yet implemented")
def test_reduce_max_op(device):
    """Test reduce_max operation."""
    pass


# =============================================================================
# Transpose Test (placeholder - to be implemented)
# =============================================================================


@pytest.mark.skip(reason="transpose not yet implemented")
def test_transpose_op(device):
    """Test transpose operation."""
    pass


# =============================================================================
# Main - for running outside pytest
# =============================================================================

if __name__ == "__main__":
    import sys

    # Check if ttnn is available
    if importlib.util.find_spec("ttnn") is None:
        print("TTNN not available - skipping tests")
        sys.exit(0)

    print("=== Reduce/Matmul Ops Test ===\n")

    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
