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
# Multi-tile Reduce Kernels (2x2 = 64x64)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_2x2_kernel(inp, scaler, out):
    """Reduce sum over 2x2 tile grid (64x64 elements)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

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
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def reduce_max_2x2_kernel(inp, scaler, out):
    """Reduce max over 2x2 tile grid (64x64 elements)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

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
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


# =============================================================================
# Multi-tile Reduce Kernels (4x4 = 128x128)
# =============================================================================


@ttl.kernel(grid=(1, 1))
def reduce_sum_4x4_kernel(inp, scaler, out):
    """Reduce sum over 4x4 tile grid (128x128 elements)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(4, 4), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

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
            tx = ttl.copy(out_blk, out[0:4, 0:4])
            tx.wait()


@ttl.kernel(grid=(1, 1))
def reduce_max_4x4_kernel(inp, scaler, out):
    """Reduce max over 4x4 tile grid (128x128 elements)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(4, 4), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)

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
            tx = ttl.copy(out_blk, out[0:4, 0:4])
            tx.wait()


# =============================================================================
# Multi-core Kernel (2x2 grid, each core processes 1 tile)
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
    """Reduce sum with 2x2 grid, each core processes 2x2 tiles (128x128 total)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

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
        row = y * 2
        col = x * 2
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[row : row + 2, col : col + 2])
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
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each tile reduces independently, sum of 32x32 ones = 1024
        expected_tile_sum = torch.tensor(32 * 32, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_tile_sum, rtol=0.1, atol=15)

    def test_reduce_sum_2x2_random(self, device):
        """Reduce sum over 2x2 tiles with random values."""
        torch.manual_seed(456)
        inp_torch = torch.randn((64, 64), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Expected: sum of first tile (top-left 32x32)
        expected_sum = inp_torch[:32, :32].float().sum().item()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(
            result[0, 0].float(),
            torch.tensor(expected_sum),
            rtol=0.15,
            atol=max(abs(expected_sum) * 0.1, 5.0),
        )

    def test_reduce_max_2x2(self, device):
        """Reduce max over 2x2 tiles with a known max value."""
        inp_torch = torch.ones((64, 64), dtype=torch.bfloat16)
        inp_torch[10, 10] = 25.0  # Max in first tile
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_max_2x2_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(result[0, 0].float(), torch.tensor(25.0), rtol=0.1, atol=0.5)


# =============================================================================
# Tests - Multi-tile (4x4)
# =============================================================================


class TestReduceMultitile4x4:
    """Tests for reduce operations with 4x4 tile grid (128x128)."""

    def test_reduce_sum_4x4_ones(self, device):
        """Reduce sum over 4x4 tiles with all ones."""
        inp_torch = torch.ones((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_4x4_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each tile reduces independently
        expected_tile_sum = torch.tensor(32 * 32, dtype=torch.float32)
        assert_allclose(result[0, 0].float(), expected_tile_sum, rtol=0.1, atol=15)

    def test_reduce_sum_4x4_random(self, device):
        """Reduce sum over 4x4 tiles with random values."""
        torch.manual_seed(789)
        inp_torch = torch.randn((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

        expected_sum = inp_torch[:32, :32].float().sum().item()

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_4x4_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        assert_allclose(
            result[0, 0].float(),
            torch.tensor(expected_sum),
            rtol=0.15,
            atol=max(abs(expected_sum) * 0.1, 5.0),
        )

    def test_reduce_max_4x4(self, device):
        """Reduce max over 4x4 tiles with a known max value."""
        inp_torch = torch.ones((128, 128), dtype=torch.bfloat16)
        inp_torch[20, 20] = 50.0  # Max in first tile
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

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
# Tests - Multi-core + Multi-tile Combined
# =============================================================================


class TestReduceMulticoreMultitile:
    """Tests for reduce with multi-core AND multi-tile (2x2 grid, 2x2 tiles each)."""

    def test_reduce_sum_multicore_multitile(self, device):
        """2x2 grid, each core processes 2x2 tiles = 128x128 total."""
        inp_torch = torch.ones((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_l1(out_torch, device)

        reduce_sum_multicore_multitile_kernel(inp, scaler, out)
        result = ttnn.to_torch(out)

        # Each tile independently sums to 1024
        expected_tile_sum = torch.tensor(32 * 32, dtype=torch.float32)
        # Check first tile of each core's region
        assert_allclose(result[0, 0].float(), expected_tile_sum, rtol=0.1, atol=15)
        assert_allclose(result[0, 64].float(), expected_tile_sum, rtol=0.1, atol=15)
        assert_allclose(result[64, 0].float(), expected_tile_sum, rtol=0.1, atol=15)
        assert_allclose(result[64, 64].float(), expected_tile_sum, rtol=0.1, atol=15)

    def test_reduce_sum_multicore_multitile_random(self, device):
        """2x2 grid with random values, verify each region independently."""
        torch.manual_seed(999)
        inp_torch = torch.randn((128, 128), dtype=torch.bfloat16)
        scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

        # Expected sums for first tile of each core's region
        expected_00 = inp_torch[:32, :32].float().sum().item()
        expected_01 = inp_torch[:32, 64:96].float().sum().item()
        expected_10 = inp_torch[64:96, :32].float().sum().item()
        expected_11 = inp_torch[64:96, 64:96].float().sum().item()

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
            atol=max(abs(expected_00) * 0.1, 5.0),
        )
        assert_allclose(
            result[0, 64].float(),
            torch.tensor(expected_01),
            rtol=tol,
            atol=max(abs(expected_01) * 0.1, 5.0),
        )
        assert_allclose(
            result[64, 0].float(),
            torch.tensor(expected_10),
            rtol=tol,
            atol=max(abs(expected_10) * 0.1, 5.0),
        )
        assert_allclose(
            result[64, 64].float(),
            torch.tensor(expected_11),
            rtol=tol,
            atol=max(abs(expected_11) * 0.1, 5.0),
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
