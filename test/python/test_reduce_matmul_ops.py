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
import ttl

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


@pytest.fixture(scope="function")
def device():
    """Open device for each test (simulator requires fresh device per test)."""
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
# Reduce Sum Kernel Template
# =============================================================================

REDUCE_SUM_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def reduce_sum_kernel(input, scaler, out):
    """Single-tile reduce sum kernel: out = reduce_sum(input, dim='scalar')."""
    in_cb = ttl.make_circular_buffer_like(input, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        in_tile = in_cb.wait()
        scaler_tile = scaler_cb.wait()
        o = out_cb.reserve()
        result = ttl.reduce_sum(in_tile, scaler_tile, o)
        o.store(result)
        in_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Load scaler first (must be ready before reduce)
        scaler_blk = scaler_cb.reserve()
        tx_s = ttl.copy(scaler[0, 0], scaler_blk)
        tx_s.wait()
        scaler_cb.push()

        in_blk = in_cb.reserve()
        tx = ttl.copy(input[0, 0], in_blk)
        tx.wait()
        in_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(input, scaler, out)
'''


def make_reduce_sum_kernel():
    """Generate reduce_sum kernel by writing to temp file and importing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_reduce_sum_"
    ) as f:
        f.write(REDUCE_SUM_KERNEL_TEMPLATE)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("reduce_sum_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "reduce_sum_kernel")


# =============================================================================
# Reduce Sum Row Kernel Template
# =============================================================================

REDUCE_SUM_ROW_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def reduce_sum_row_kernel(input, scaler, out):
    """Single-tile reduce sum kernel: row reduction (sum across columns)."""
    in_cb = ttl.make_circular_buffer_like(input, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        in_tile = in_cb.wait()
        scaler_tile = scaler_cb.wait()
        o = out_cb.reserve()
        result = ttl.reduce_sum(in_tile, scaler_tile, o, dim="row")
        o.store(result)
        in_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        scaler_blk = scaler_cb.reserve()
        tx_s = ttl.copy(scaler[0, 0], scaler_blk)
        tx_s.wait()
        scaler_cb.push()

        in_blk = in_cb.reserve()
        tx = ttl.copy(input[0, 0], in_blk)
        tx.wait()
        in_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(input, scaler, out)
'''


def make_reduce_sum_row_kernel():
    """Generate reduce_sum row kernel by writing to temp file and importing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_reduce_sum_row_"
    ) as f:
        f.write(REDUCE_SUM_ROW_KERNEL_TEMPLATE)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("reduce_sum_row_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "reduce_sum_row_kernel")


# =============================================================================
# Reduce Sum Col Kernel Template
# =============================================================================

REDUCE_SUM_COL_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def reduce_sum_col_kernel(input, scaler, out):
    """Single-tile reduce sum kernel: column reduction (sum across rows)."""
    in_cb = ttl.make_circular_buffer_like(input, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        in_tile = in_cb.wait()
        scaler_tile = scaler_cb.wait()
        o = out_cb.reserve()
        result = ttl.reduce_sum(in_tile, scaler_tile, o, dim="col")
        o.store(result)
        in_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        scaler_blk = scaler_cb.reserve()
        tx_s = ttl.copy(scaler[0, 0], scaler_blk)
        tx_s.wait()
        scaler_cb.push()

        in_blk = in_cb.reserve()
        tx = ttl.copy(input[0, 0], in_blk)
        tx.wait()
        in_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(input, scaler, out)
'''


def make_reduce_sum_col_kernel():
    """Generate reduce_sum col kernel by writing to temp file and importing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_reduce_sum_col_"
    ) as f:
        f.write(REDUCE_SUM_COL_KERNEL_TEMPLATE)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("reduce_sum_col_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "reduce_sum_col_kernel")


# =============================================================================
# Reduce Max Kernel Template
# =============================================================================

REDUCE_MAX_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def reduce_max_kernel(input, scaler, out):
    """Single-tile reduce max kernel: out = reduce_max(input, dim='scalar')."""
    in_cb = ttl.make_circular_buffer_like(input, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        in_tile = in_cb.wait()
        scaler_tile = scaler_cb.wait()
        o = out_cb.reserve()
        result = ttl.reduce_max(in_tile, scaler_tile, o)
        o.store(result)
        in_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Load scaler first (must be ready before reduce)
        scaler_blk = scaler_cb.reserve()
        tx_s = ttl.copy(scaler[0, 0], scaler_blk)
        tx_s.wait()
        scaler_cb.push()

        in_blk = in_cb.reserve()
        tx = ttl.copy(input[0, 0], in_blk)
        tx.wait()
        in_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(input, scaler, out)
'''


def make_reduce_max_kernel():
    """Generate reduce_max kernel by writing to temp file and importing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_reduce_max_"
    ) as f:
        f.write(REDUCE_MAX_KERNEL_TEMPLATE)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("reduce_max_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "reduce_max_kernel")


# =============================================================================
# Transpose Kernel Template
# =============================================================================

TRANSPOSE_KERNEL_TEMPLATE = '''
import ttl

@ttl.kernel(grid=(1, 1))
def transpose_kernel(input, out):
    """Single-tile transpose kernel: out = transpose(input)."""
    in_cb = ttl.make_circular_buffer_like(input, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        in_tile = in_cb.wait()
        o = out_cb.reserve()
        result = ttl.transpose(in_tile, o)
        o.store(result)
        in_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        in_blk = in_cb.reserve()
        tx = ttl.copy(input[0, 0], in_blk)
        tx.wait()
        in_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(input, out)
'''


def make_transpose_kernel():
    """Generate transpose kernel by writing to temp file and importing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="kernel_transpose_"
    ) as f:
        f.write(TRANSPOSE_KERNEL_TEMPLATE)
        temp_path = f.name

    spec = importlib.util.spec_from_file_location("transpose_kernel_module", temp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "transpose_kernel")


# =============================================================================
# Reduce Tests
# =============================================================================


def test_reduce_sum_op(device):
    """Test reduce_sum operation (scalar reduction)."""
    kernel = make_reduce_sum_kernel()

    # Single tile: 32x32, sum to scalar (result is 32x32 with sum in [0,0])
    in_torch = torch.randn((32, 32), dtype=torch.bfloat16)
    scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)  # Scaler with 1.0 for simple sum
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result - scalar sum
    expected_sum = in_torch.float().sum().item()

    # Create device tensors
    input_tensor = ttnn.from_torch(
        in_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scaler_tensor = ttnn.from_torch(
        scaler_torch,
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
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    scaler_tensor = ttnn.to_memory_config(scaler_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    kernel(input_tensor, scaler_tensor, out)

    # Get result - scalar sum should be in position [0,0]
    result = ttnn.to_torch(out)
    result_sum = result[0, 0].float().item()

    # Compare with relaxed tolerance for reduction
    assert abs(result_sum - expected_sum) < abs(expected_sum) * 0.1 + 1.0, \
        f"reduce_sum mismatch: got {result_sum}, expected {expected_sum}"


def test_reduce_sum_row_op(device):
    """Test reduce_sum with dim='row' (sum across columns for each row)."""
    kernel = make_reduce_sum_row_kernel()

    # Single tile: 32x32, row reduction produces column vector in column 0
    in_torch = torch.randn((32, 32), dtype=torch.bfloat16)
    scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Row reduction: sum each row -> values in column 0
    expected_row_sums = in_torch.float().sum(dim=1)  # Shape: [32]

    # Create device tensors
    input_tensor = ttnn.from_torch(
        in_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scaler_tensor = ttnn.from_torch(
        scaler_torch,
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
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    scaler_tensor = ttnn.to_memory_config(scaler_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    kernel(input_tensor, scaler_tensor, out)

    # Get result - row sums should be in column 0
    result = ttnn.to_torch(out)
    result_row_sums = result[:, 0].float()

    # Compare with relaxed tolerance
    max_error = (result_row_sums - expected_row_sums).abs().max().item()
    rel_error = max_error / (expected_row_sums.abs().max().item() + 1e-6)
    assert rel_error < 0.1, \
        f"reduce_sum row mismatch: relative error {rel_error}"


def test_reduce_sum_col_op(device):
    """Test reduce_sum with dim='col' (sum across rows for each column)."""
    kernel = make_reduce_sum_col_kernel()

    # Single tile: 32x32, col reduction produces row vector in row 0
    in_torch = torch.randn((32, 32), dtype=torch.bfloat16)
    scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Col reduction: sum each column -> values in row 0
    expected_col_sums = in_torch.float().sum(dim=0)  # Shape: [32]

    # Create device tensors
    input_tensor = ttnn.from_torch(
        in_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scaler_tensor = ttnn.from_torch(
        scaler_torch,
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
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    scaler_tensor = ttnn.to_memory_config(scaler_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    kernel(input_tensor, scaler_tensor, out)

    # Get result - col sums should be in row 0
    result = ttnn.to_torch(out)
    result_col_sums = result[0, :].float()

    # Compare with relaxed tolerance
    max_error = (result_col_sums - expected_col_sums).abs().max().item()
    rel_error = max_error / (expected_col_sums.abs().max().item() + 1e-6)
    assert rel_error < 0.1, \
        f"reduce_sum col mismatch: relative error {rel_error}"


# @pytest.mark.skip(reason="reduce_max uses tensix_execute_gmpool not implemented in simulator - test hangs")
def test_reduce_max_op(device):
    """Test reduce_max operation (scalar reduction)."""
    kernel = make_reduce_max_kernel()

    # Single tile: 32x32, max to scalar (result is 32x32 with max in [0,0])
    in_torch = torch.randn((32, 32), dtype=torch.bfloat16)
    scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)  # Scaler with 1.0
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result - scalar max
    expected_max = in_torch.float().max().item()

    # Create device tensors
    input_tensor = ttnn.from_torch(
        in_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scaler_tensor = ttnn.from_torch(
        scaler_torch,
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
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    scaler_tensor = ttnn.to_memory_config(scaler_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    kernel(input_tensor, scaler_tensor, out)

    # Get result - scalar max should be in position [0,0]
    result = ttnn.to_torch(out)
    result_max = result[0, 0].float().item()

    # Compare with relaxed tolerance for reduction
    assert abs(result_max - expected_max) < abs(expected_max) * 0.1 + 0.1, \
        f"reduce_max mismatch: got {result_max}, expected {expected_max}"


# =============================================================================
# Transpose Test
# =============================================================================


def test_transpose_op(device):
    """Test transpose operation."""
    kernel = make_transpose_kernel()

    # Single tile: 32x32 transpose
    in_torch = torch.randn((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Compute expected result
    expected = in_torch.T.contiguous()

    # Create device tensors
    input_tensor = ttnn.from_torch(
        in_torch,
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
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Run kernel
    kernel(input_tensor, out)

    # Get result
    result = ttnn.to_torch(out)

    # Compare
    assert_allclose(result.float(), expected.float(), rtol=1e-3, atol=1e-3)


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
