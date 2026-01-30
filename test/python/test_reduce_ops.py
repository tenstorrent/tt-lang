# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTL reduce operations.

Tests reduce_sum and reduce_max operations with various reduction dimensions.
The reduce ops read from input CB, apply reduction, and write to output CB.

Test cases:
1. Scalar reduction (dims=[0, 1]): reduce entire tile to single value
2. Random inputs with PyTorch reference
3. DRAM memory configuration
4. Multi-tile (2x2 and 4x4)
5. Multi-core (2x2 grid)
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1, to_dram

import ttl


# =============================================================================
# Single-tile Reduce Kernels
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_scalar_kernel(inp, scaler, out):
    """Reduce sum: sum all elements in tile to a scalar (dims=[0, 1])."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0, 0], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def reduce_max_scalar_kernel(inp, scaler, out):
    """Reduce max: find max element in tile (dims=[0, 1])."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_max(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0, 0], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


# =============================================================================
# Row Reduction Kernels (dims=[0])
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_row_kernel(inp, scaler, out):
    """Reduce sum across columns for each row (dims=[0]).

    Result: each row's sum appears in column 0 of that row.
    """
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0, 0], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


# =============================================================================
# Col Reduction Kernels (dims=[1])
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_col_kernel(inp, scaler, out):
    """Reduce sum across rows for each column (dims=[1]).

    Result: each column's sum appears in row 0 of that column.
    """
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0, 0], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


# =============================================================================
# Multi-tile Row/Col Reduce Kernels (2x2 = 64x64)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_row_2x2_kernel(inp, scaler, out):
    """Reduce sum rows over 2x2 tile grid (dims=[0]) -> (2,1) output."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0:2, 0:2], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0:2, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def reduce_sum_col_2x2_kernel(inp, scaler, out):
    """Reduce sum cols over 2x2 tile grid (dims=[1]) -> (1,2) output."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0:2, 0:2], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0:2])
            tx.wait()


# =============================================================================
# Multi-tile Reduce Kernels (2x2 = 64x64) - Scalar
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_2x2_kernel(inp, scaler, out):
    """Reduce sum over 2x2 tile grid (64x64 elements) -> scalar."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0:2, 0:2], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def reduce_max_2x2_kernel(inp, scaler, out):
    """Reduce max over 2x2 tile grid (64x64 elements) -> scalar."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_max(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0:2, 0:2], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


# =============================================================================
# Multi-tile Reduce Kernels (4x4 = 128x128)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_4x4_kernel(inp, scaler, out):
    """Reduce sum over 4x4 tile grid (128x128 elements) -> scalar."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(4, 4), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0:4, 0:4], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def reduce_max_4x4_kernel(inp, scaler, out):
    """Reduce max over 4x4 tile grid (128x128 elements) -> scalar."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(4, 4), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_max(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[0:4, 0:4], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


# =============================================================================
# Multi-core Row/Col Kernels (2x2 grid, each core processes 1 tile)
# =============================================================================


@ttl.kernel(grid=(2, 2))
def reduce_sum_row_multicore_kernel(inp, scaler, out):
    """Reduce sum rows with 2x2 multicore grid (dims=[0])."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[y, x], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[y, x], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


@ttl.kernel(grid=(2, 2))
def reduce_sum_col_multicore_kernel(inp, scaler, out):
    """Reduce sum cols with 2x2 multicore grid (dims=[1])."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[y, x], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[y, x], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


# =============================================================================
# Multi-core Kernel (2x2 grid, each core processes 1 tile) - Scalar
# =============================================================================


@ttl.kernel(grid=(2, 2))
def reduce_sum_multicore_kernel(inp, scaler, out):
    """Reduce sum with 2x2 multicore grid. Each core reduces its own tile."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[y, x], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[y, x], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


# =============================================================================
# Multi-core + Multi-tile Kernel (2x2 grid, each core processes 2x2 tiles)
# =============================================================================


@ttl.kernel(grid=(2, 2))
def reduce_sum_multicore_multitile_kernel(inp, scaler, out):
    """Reduce sum with 2x2 grid, each core processes 2x2 tiles -> 1 tile output."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(i, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        row = y * 2
        col = x * 2
        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[row : row + 2, col : col + 2], inp_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


# =============================================================================
# Tests - Single Tile
# =============================================================================


class TestReduceSumScalar:
    """Tests for reduce_sum with dims=[0, 1] (scalar reduction)."""

    def test_reduce_sum_ones(self, device):
        """Sum of all ones: should be 32*32 = 1024."""
        inp_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        expected_sum = torch.tensor(32 * 32, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.1, atol=10)

    def test_reduce_sum_values(self, device):
        """Sum of 2.0 values: should be 32*32*2 = 2048."""
        inp_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        expected_sum = torch.tensor(32 * 32 * 2, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.1, atol=20)

    def test_reduce_sum_random(self, device):
        """Sum of random values against PyTorch reference."""
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        expected_sum = inp_torch.float().sum().item()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(
            result[0, 0].float(),
            torch.tensor(expected_sum),
            rtol=0.15,
            atol=max(abs(expected_sum) * 0.1, 1.0),
        )


class TestReduceMaxScalar:
    """Tests for reduce_max with dims=[0, 1] (scalar reduction)."""

    def test_reduce_max_uniform(self, device):
        """Max of uniform values: should be that value."""
        inp_torch = torch.full((32, 32), 5.0, dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(result[0, 0].float(), torch.tensor(5.0), rtol=0.1, atol=0.1)

    def test_reduce_max_single_max(self, device):
        """Max with one element larger: should find that max."""
        inp_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        inp_torch[15, 15] = 10.0
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(result[0, 0].float(), torch.tensor(10.0), rtol=0.1, atol=0.5)

    def test_reduce_max_random(self, device):
        """Max of random values against PyTorch reference."""
        torch.manual_seed(123)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        expected_max = inp_torch.float().max().item()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(
            result[0, 0].float(),
            torch.tensor(expected_max),
            rtol=0.1,
            atol=max(abs(expected_max) * 0.1, 0.5),
        )


# =============================================================================
# Tests - Row Reduction
# =============================================================================


class TestReduceRow:
    """Tests for reduce_sum with dims=[0] (row reduction - sum across columns)."""

    def test_reduce_sum_row_ones(self, device):
        """Row reduction of all ones: each row sums to 32."""
        inp_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_row_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Row sums appear in column 0
        expected_row_sum = torch.tensor(32.0, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_row_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[15, 0].float(), expected_row_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[31, 0].float(), expected_row_sum, rtol=0.1, atol=1.0)

    def test_reduce_sum_row_random(self, device):
        """Row reduction with random values against PyTorch reference."""
        torch.manual_seed(111)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: sum of each row
        expected_row_sums = inp_torch.float().sum(dim=1)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_row_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check column 0 contains row sums
        result_row_sums = result[:, 0].float()
        max_error = (result_row_sums - expected_row_sums).abs().max().item()
        rel_error = max_error / (expected_row_sums.abs().max().item() + 1e-6)
        assert rel_error < 0.15, f"Row reduction mismatch: relative error {rel_error}"


# =============================================================================
# Tests - Col Reduction
# =============================================================================


class TestReduceCol:
    """Tests for reduce_sum with dims=[1] (col reduction - sum across rows)."""

    def test_reduce_sum_col_ones(self, device):
        """Col reduction of all ones: each column sums to 32."""
        inp_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_col_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Col sums appear in row 0
        expected_col_sum = torch.tensor(32.0, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_col_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[0, 15].float(), expected_col_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[0, 31].float(), expected_col_sum, rtol=0.1, atol=1.0)

    def test_reduce_sum_col_random(self, device):
        """Col reduction with random values against PyTorch reference."""
        torch.manual_seed(222)
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: sum of each column
        expected_col_sums = inp_torch.float().sum(dim=0)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_col_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Check row 0 contains column sums
        result_col_sums = result[0, :].float()
        max_error = (result_col_sums - expected_col_sums).abs().max().item()
        rel_error = max_error / (expected_col_sums.abs().max().item() + 1e-6)
        assert rel_error < 0.15, f"Col reduction mismatch: relative error {rel_error}"


# =============================================================================
# Tests - DRAM Memory
# =============================================================================


class TestReduceDRAM:
    """Tests for reduce operations with DRAM memory."""

    def test_reduce_sum_dram(self, device):
        """Reduce sum with DRAM tensors."""
        inp_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        scaler = to_dram(scaler_torch, device)
        out = to_dram(out_torch, device)

        reduce_sum_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        expected_sum = torch.tensor(32 * 32 * 3, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.1, atol=30)

    def test_reduce_max_dram(self, device):
        """Reduce max with DRAM tensors."""
        inp_torch = torch.full((32, 32), 7.0, dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        scaler = to_dram(scaler_torch, device)
        out = to_dram(out_torch, device)

        reduce_max_scalar_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(result[0, 0].float(), torch.tensor(7.0), rtol=0.1, atol=0.5)


# =============================================================================
# Tests - Multi-tile (2x2)
# =============================================================================


class TestReduceMultitile2x2:
    """Tests for reduce operations with 2x2 tile grid (64x64)."""

    def test_reduce_sum_2x2_ones(self, device):
        """Reduce sum over 2x2 tiles with all ones."""
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # All 4 tiles accumulated: 64*64 = 4096
        expected_sum = torch.tensor(64 * 64, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.1, atol=50)

    def test_reduce_sum_2x2_random(self, device):
        """Reduce sum over 2x2 tiles with random values."""
        torch.manual_seed(456)
        inp_torch = torch.randn((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: sum of all 64x64 elements (all 4 tiles)
        expected_sum = inp_torch.float().sum().item()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(
            result[0, 0].float(),
            torch.tensor(expected_sum),
            rtol=0.15,
            atol=max(abs(expected_sum) * 0.1, 10.0),
        )

    def test_reduce_max_2x2(self, device):
        """Reduce max over 2x2 tiles with a known max value."""
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        inp_torch[10, 10] = 25.0  # Max in first tile
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(result[0, 0].float(), torch.tensor(25.0), rtol=0.1, atol=0.5)

    def test_reduce_sum_row_2x2(self, device):
        """Row reduction over 2x2 tiles: each row sums across 64 columns."""
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_row_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each row sums 64 columns (2 tiles of 32)
        expected_row_sum = torch.tensor(64.0, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_row_sum, rtol=0.1, atol=2.0)
        assert_allclose(result[15, 0].float(), expected_row_sum, rtol=0.1, atol=2.0)

    def test_reduce_sum_col_2x2(self, device):
        """Col reduction over 2x2 tiles: each col sums across 64 rows."""
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_col_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each column sums 64 rows (2 tiles of 32)
        expected_col_sum = torch.tensor(64.0, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_col_sum, rtol=0.1, atol=2.0)
        assert_allclose(result[0, 15].float(), expected_col_sum, rtol=0.1, atol=2.0)


# =============================================================================
# Tests - Multi-tile (4x4)
# =============================================================================


class TestReduceMultitile4x4:
    """Tests for reduce operations with 4x4 tile grid (128x128)."""

    def test_reduce_sum_4x4_ones(self, device):
        """Reduce sum over 4x4 tiles with all ones."""
        inp_torch = torch.ones((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_4x4_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # All 16 tiles accumulated: 128*128 = 16384
        expected_sum = torch.tensor(128 * 128, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.1, atol=200)

    def test_reduce_sum_4x4_random(self, device):
        """Reduce sum over 4x4 tiles with random values."""
        torch.manual_seed(789)
        inp_torch = torch.randn((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: sum of all 128x128 elements (all 16 tiles)
        expected_sum = inp_torch.float().sum().item()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_4x4_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(
            result[0, 0].float(),
            torch.tensor(expected_sum),
            rtol=0.15,
            atol=max(abs(expected_sum) * 0.1, 20.0),
        )

    def test_reduce_max_4x4(self, device):
        """Reduce max over 4x4 tiles with a known max value."""
        inp_torch = torch.ones((128, 128), dtype=torch.bfloat16)
        inp_torch[20, 20] = 50.0  # Max in first tile
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_4x4_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(result[0, 0].float(), torch.tensor(50.0), rtol=0.1, atol=0.5)


# =============================================================================
# Tests - Multi-core
# =============================================================================


class TestReduceMulticore:
    """Tests for reduce operations with multi-core (2x2 grid)."""

    def test_reduce_sum_multicore(self, device):
        """Each core reduces its own tile, results placed in respective outputs."""
        # 2x2 tiles = 64x64 total
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_multicore_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        expected_tile_sum = torch.tensor(32 * 32, dtype=torch.float32)
        # Each tile's [0,0] should have the sum
        assert_allclose(result[0, 0].float(), expected_tile_sum, rtol=0.1, atol=15)
        assert_allclose(result[0, 32].float(), expected_tile_sum, rtol=0.1, atol=15)
        assert_allclose(result[32, 0].float(), expected_tile_sum, rtol=0.1, atol=15)
        assert_allclose(result[32, 32].float(), expected_tile_sum, rtol=0.1, atol=15)

    def test_reduce_sum_multicore_distinct(self, device):
        """Each core reduces distinct values."""
        inp_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        # Set distinct values in each tile
        inp_torch[:32, :32] = 1.0  # Tile [0,0]: sum = 1024
        inp_torch[:32, 32:64] = 2.0  # Tile [0,1]: sum = 2048
        inp_torch[32:64, :32] = 3.0  # Tile [1,0]: sum = 3072
        inp_torch[32:64, 32:64] = 4.0  # Tile [1,1]: sum = 4096

        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_multicore_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(result[0, 0].float(), torch.tensor(1024.0), rtol=0.1, atol=15)
        assert_allclose(result[0, 32].float(), torch.tensor(2048.0), rtol=0.1, atol=25)
        assert_allclose(result[32, 0].float(), torch.tensor(3072.0), rtol=0.1, atol=35)
        assert_allclose(result[32, 32].float(), torch.tensor(4096.0), rtol=0.1, atol=50)


# =============================================================================
# Tests - Multi-core Row/Col
# =============================================================================


class TestReduceMulticoreRowCol:
    """Tests for row/col reduce operations with multi-core (2x2 grid)."""

    def test_reduce_sum_row_multicore(self, device):
        """Row reduction with 2x2 multicore grid."""
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_row_multicore_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each tile's rows sum to 32
        expected_row_sum = torch.tensor(32.0, dtype=torch.float32)
        # Check column 0 of each tile
        assert_allclose(result[0, 0].float(), expected_row_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[0, 32].float(), expected_row_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[32, 0].float(), expected_row_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[32, 32].float(), expected_row_sum, rtol=0.1, atol=1.0)

    def test_reduce_sum_col_multicore(self, device):
        """Col reduction with 2x2 multicore grid."""
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_col_multicore_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each tile's cols sum to 32
        expected_col_sum = torch.tensor(32.0, dtype=torch.float32)
        # Check row 0 of each tile
        assert_allclose(result[0, 0].float(), expected_col_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[0, 32].float(), expected_col_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[32, 0].float(), expected_col_sum, rtol=0.1, atol=1.0)
        assert_allclose(result[32, 32].float(), expected_col_sum, rtol=0.1, atol=1.0)


# =============================================================================
# Tests - Multi-core + Multi-tile Combined
# =============================================================================


class TestReduceMulticoreMultitile:
    """Tests for reduce with multi-core AND multi-tile (2x2 grid, 2x2 tiles each)."""

    def test_reduce_sum_multicore_multitile(self, device):
        """2x2 grid, each core processes 2x2 tiles = 128x128 total."""
        inp_torch = torch.ones((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_multicore_multitile_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each core reduces 2x2 tiles = 64x64 elements = 4096
        expected_sum = torch.tensor(64 * 64, dtype=torch.float32)
        # Check each core's output position
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.1, atol=50)
        assert_allclose(result[0, 32].float(), expected_sum, rtol=0.1, atol=50)
        assert_allclose(result[32, 0].float(), expected_sum, rtol=0.1, atol=50)
        assert_allclose(result[32, 32].float(), expected_sum, rtol=0.1, atol=50)

    def test_reduce_sum_multicore_multitile_random(self, device):
        """2x2 grid with random values, verify each region independently."""
        torch.manual_seed(999)
        inp_torch = torch.randn((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Expected sums for each core's full 2x2 tile region (64x64 elements)
        expected_00 = inp_torch[:64, :64].float().sum().item()
        expected_01 = inp_torch[:64, 64:128].float().sum().item()
        expected_10 = inp_torch[64:128, :64].float().sum().item()
        expected_11 = inp_torch[64:128, 64:128].float().sum().item()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_multicore_multitile_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        tol = 0.15
        assert_allclose(
            result[0, 0].float(),
            torch.tensor(expected_00),
            rtol=tol,
            atol=max(abs(expected_00) * 0.1, 10.0),
        )
        assert_allclose(
            result[0, 32].float(),
            torch.tensor(expected_01),
            rtol=tol,
            atol=max(abs(expected_01) * 0.1, 10.0),
        )
        assert_allclose(
            result[32, 0].float(),
            torch.tensor(expected_10),
            rtol=tol,
            atol=max(abs(expected_10) * 0.1, 10.0),
        )
        assert_allclose(
            result[32, 32].float(),
            torch.tensor(expected_11),
            rtol=tol,
            atol=max(abs(expected_11) * 0.1, 10.0),
        )


# =============================================================================
# Combined Operations: Bcast + Reduce (multicore, multitile)
# =============================================================================


@ttl.kernel(grid=(2, 2))
def bcast_then_reduce_kernel(inp, bcast_in, scaler, out):
    """Combine bcast and reduce: broadcast input, then reduce the result.

    Pattern: reduce_sum(broadcast(bcast_in) + inp)
    - bcast_in: scalar to broadcast to full tile
    - inp: full tile input
    - Result: sum of (broadcast_value + inp) for all tiles in each core's region
    """
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    bcast_cb = ttl.make_circular_buffer_like(bcast_in, shape=(2, 2), buffer_factor=2)
    bcast_out_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    add_out_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        # Stage 1: Broadcast
        with bcast_cb.wait() as b, bcast_out_cb.reserve() as b_out:
            result = ttl.math.broadcast(b, b_out, dims=[0, 1])
            b_out.store(result)

        # Stage 2: Add broadcasted value to input
        with (
            inp_cb.wait() as i,
            bcast_out_cb.wait() as b_val,
            add_out_cb.reserve() as a_out,
        ):
            added = i + b_val
            a_out.store(added)

        # Stage 3: Reduce the add result across all 2x2 tiles
        with add_out_cb.wait() as a_in, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(a_in, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        row = y * 2
        col = x * 2

        with inp_cb.reserve() as inp_blk:
            tx = ttl.copy(inp[row : row + 2, col : col + 2], inp_blk)
            tx.wait()

        with bcast_cb.reserve() as bcast_blk:
            tx = ttl.copy(bcast_in[row : row + 2, col : col + 2], bcast_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


class TestBcastThenReduce:
    """Tests for combined broadcast + reduce operations."""

    def test_bcast_then_reduce_multicore_multitile(self, device):
        """Broadcast scalar then reduce: 2x2 grid, 2x2 tiles each.

        inp = all 1.0
        bcast_in = scalar 2.0 (will be broadcast to full tiles)
        Each core: sums (1.0 + 2.0) * 64 * 64 = 12288 (2x2 tiles = 64x64 elements)
        """
        inp_torch = torch.ones((128, 128), dtype=torch.bfloat16)

        # Scalar broadcast input: [0,0] of each tile has 2.0
        bcast_torch = torch.zeros((128, 128), dtype=torch.bfloat16)
        bcast_torch[0, 0] = 2.0
        bcast_torch[0, 32] = 2.0
        bcast_torch[32, 0] = 2.0
        bcast_torch[32, 32] = 2.0
        bcast_torch[0, 64] = 2.0
        bcast_torch[0, 96] = 2.0
        bcast_torch[32, 64] = 2.0
        bcast_torch[32, 96] = 2.0
        bcast_torch[64, 0] = 2.0
        bcast_torch[64, 32] = 2.0
        bcast_torch[96, 0] = 2.0
        bcast_torch[96, 32] = 2.0
        bcast_torch[64, 64] = 2.0
        bcast_torch[64, 96] = 2.0
        bcast_torch[96, 64] = 2.0
        bcast_torch[96, 96] = 2.0

        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        bcast_in = to_l1(bcast_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        bcast_then_reduce_kernel(inp, bcast_in, scaler, out)
        result = ttnn.to_torch(out)

        # Each core reduces 2x2 tiles (64x64 elements): (1.0 + 2.0) * 64 * 64 = 12288
        expected_sum = torch.tensor(3.0 * 64 * 64, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.1, atol=150)
        assert_allclose(result[0, 32].float(), expected_sum, rtol=0.1, atol=150)
        assert_allclose(result[32, 0].float(), expected_sum, rtol=0.1, atol=150)
        assert_allclose(result[32, 32].float(), expected_sum, rtol=0.1, atol=150)


# =============================================================================
# Combined Operations: Matmul + Reduce (multicore, multitile)
# =============================================================================


@ttl.kernel(grid=(2, 2))
def matmul_then_reduce_kernel(a, b, scaler, out):
    """Combine matmul and reduce: matmul then reduce the result.

    Pattern: reduce_sum(matmul(a, b))
    Each core does: C = A @ B (2x2 tiles), then sum = reduce_sum(C) (1 tile)
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(2, 2), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(2, 2), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        # Stage 1: Matmul (2x2 tiles output)
        with a_cb.wait() as av, b_cb.wait() as bv, c_cb.reserve() as cv:
            result = ttl.math.matmul(av, bv, cv)
            cv.store(result)

        # Stage 2: Reduce all 4 tiles to scalar
        with c_cb.wait() as c, scaler_cb.wait() as s, out_cb.reserve() as o:
            result = ttl.math.reduce_sum(c, s, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        row = y * 2
        col = x * 2

        with a_cb.reserve() as a_blk:
            tx = ttl.copy(a[row : row + 2, col : col + 2], a_blk)
            tx.wait()

        with b_cb.reserve() as b_blk:
            tx = ttl.copy(b[row : row + 2, col : col + 2], b_blk)
            tx.wait()

        with scaler_cb.reserve() as scaler_blk:
            tx = ttl.copy(scaler[0, 0], scaler_blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()


class TestMatmulThenReduce:
    """Tests for combined matmul + reduce operations."""

    def test_matmul_then_reduce_multicore_multitile(self, device):
        """Matmul then reduce: 2x2 grid, 2x2 tiles each.

        a = all 0.1, b = all 0.1
        For 2x2 tile blocks (K=2 tiles): tile matmul accumulates over K tiles
        Each tile matmul element: 0.1 * 0.1 * 32 * 2 = 0.64 per element
        Matmul produces 2x2 tiles = 64x64 elements
        reduce_sum of 64x64: 0.64 * 64 * 64 = 2621.44
        """
        a_torch = torch.full((128, 128), 0.1, dtype=torch.bfloat16)
        b_torch = torch.full((128, 128), 0.1, dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # K=2 tiles (64 elements), each tile matmul gives 0.1*0.1*32*2 = 0.64 per element
        # Reduce 2x2 tiles (64x64 elements): 0.64 * 64 * 64 = 2621.44
        expected_sum = torch.tensor(0.1 * 0.1 * 32 * 2 * 64 * 64, dtype=torch.float32)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        matmul_then_reduce_kernel(a, b, scaler, out)
        result = ttnn.to_torch(out)

        # Relaxed tolerance for matmul + reduce chain
        assert_allclose(result[0, 0].float(), expected_sum, rtol=0.15, atol=200)
        assert_allclose(result[0, 32].float(), expected_sum, rtol=0.15, atol=200)
        assert_allclose(result[32, 0].float(), expected_sum, rtol=0.15, atol=200)
        assert_allclose(result[32, 32].float(), expected_sum, rtol=0.15, atol=200)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
