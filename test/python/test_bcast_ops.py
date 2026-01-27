# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTL broadcast operations.

Tests the tile_bcast op which broadcasts a row/col/scalar tile to a full tile.
Also tests composition patterns like (a * b) + bcast(c).
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttlang_test_utils import assert_allclose, to_l1

import ttl


# =============================================================================
# Helper to create input tensors for broadcast tests
# =============================================================================


def create_row_bcast_input(value: float, dtype=torch.bfloat16):
    """Create a tile with first row filled, rest zeros (for row broadcast)."""
    t = torch.zeros((32, 32), dtype=dtype)
    t[0, :] = value  # Fill first row
    return t


def create_col_bcast_input(value: float, dtype=torch.bfloat16):
    """Create a tile with first column filled, rest zeros (for col broadcast)."""
    t = torch.zeros((32, 32), dtype=dtype)
    t[:, 0] = value  # Fill first column
    return t


def create_scalar_bcast_input(value: float, dtype=torch.bfloat16):
    """Create a tile with single value at [0,0], rest zeros (for scalar broadcast)."""
    t = torch.zeros((32, 32), dtype=dtype)
    t[0, 0] = value  # Fill top-left element
    return t


def expected_bcast_result(value: float, dtype=torch.bfloat16):
    """Expected result: full tile filled with the broadcast value."""
    return torch.full((32, 32), value, dtype=dtype)


# =============================================================================
# Simple Bcast Kernel - bcast as first op in compute
# =============================================================================


@ttl.kernel(grid=(1, 1))
def bcast_row_kernel(inp, out):
    """Broadcast row tile to full tile (bcast as first op in compute)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.bcast(i, o, "row")
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx = ttl.copy(inp[0, 0], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


@ttl.kernel(grid=(1, 1))
def bcast_col_kernel(inp, out):
    """Broadcast column tile to full tile."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.bcast(i, o, "col")
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx = ttl.copy(inp[0, 0], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


@ttl.kernel(grid=(1, 1))
def bcast_scalar_kernel(inp, out):
    """Broadcast scalar tile to full tile."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.bcast(i, o, "scalar")
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx = ttl.copy(inp[0, 0], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


# =============================================================================
# Composition Pattern: (a * b) + bcast(c)
# Tests bcast as first op with subsequent DST operations
# =============================================================================


@ttl.kernel(grid=(1, 1))
def mul_add_bcast_kernel(a, b, c, out):
    """Compute (a * b) + bcast(c) where c is a row-broadcast tile.

    This tests the composition pattern where bcast reads from CB first,
    then we do DST-to-DST ops (mul, add).

    NOTE: Currently we need two stages because bcast reads from CB while
    add/mul read from DST. It would be trivial to allow CB-reading ops
    (bcast, reduce, transpose) as the FIRST op only in a fused compute block.
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    c_bcast_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        # Stage 1: Bcast c and store to intermediate CB
        with c_cb.wait() as c_tile, c_bcast_cb.reserve() as c_out:
            c_bcast = ttl.bcast(c_tile, c_out, "row")
            c_out.store(c_bcast)

        # Stage 2: Compute (a * b) + c_bcast
        with (
            a_cb.wait() as a_tile,
            b_cb.wait() as b_tile,
            c_bcast_cb.wait() as c_bcast_tile,
            out_cb.reserve() as o,
        ):
            ab = a_tile * b_tile
            result = ab + c_bcast_tile
            o.store(result)

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

        c_blk = c_cb.reserve()
        tx_c = ttl.copy(c[0, 0], c_blk)
        tx_c.wait()
        c_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


# =============================================================================
# Multi-tile Bcast Kernels (2x2 tile grid)
# =============================================================================


def create_multitile_row_bcast_input(values, dtype=torch.bfloat16):
    """Create a 64x64 tensor (2x2 tiles) with first row of each tile filled."""
    t = torch.zeros((64, 64), dtype=dtype)
    # Tile (0,0) at rows 0-31, cols 0-31
    t[0, 0:32] = values[0]
    # Tile (0,1) at rows 0-31, cols 32-63
    t[0, 32:64] = values[1]
    # Tile (1,0) at rows 32-63, cols 0-31
    t[32, 0:32] = values[2]
    # Tile (1,1) at rows 32-63, cols 32-63
    t[32, 32:64] = values[3]
    return t


def create_multitile_col_bcast_input(values, dtype=torch.bfloat16):
    """Create a 64x64 tensor (2x2 tiles) with first col of each tile filled."""
    t = torch.zeros((64, 64), dtype=dtype)
    # Tile (0,0)
    t[0:32, 0] = values[0]
    # Tile (0,1)
    t[0:32, 32] = values[1]
    # Tile (1,0)
    t[32:64, 0] = values[2]
    # Tile (1,1)
    t[32:64, 32] = values[3]
    return t


def create_multitile_scalar_bcast_input(values, dtype=torch.bfloat16):
    """Create a 64x64 tensor (2x2 tiles) with [0,0] of each tile filled."""
    t = torch.zeros((64, 64), dtype=dtype)
    t[0, 0] = values[0]  # Tile (0,0)
    t[0, 32] = values[1]  # Tile (0,1)
    t[32, 0] = values[2]  # Tile (1,0)
    t[32, 32] = values[3]  # Tile (1,1)
    return t


def expected_multitile_bcast_result(values, dtype=torch.bfloat16):
    """Expected result: each tile filled with its corresponding value."""
    t = torch.zeros((64, 64), dtype=dtype)
    t[0:32, 0:32] = values[0]  # Tile (0,0)
    t[0:32, 32:64] = values[1]  # Tile (0,1)
    t[32:64, 0:32] = values[2]  # Tile (1,0)
    t[32:64, 32:64] = values[3]  # Tile (1,1)
    return t


@ttl.kernel(grid=(1, 1))
def bcast_row_multitile_kernel(inp, out):
    """Broadcast row tiles to full tiles (2x2 grid = 4 tiles)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.bcast(i, o, "row")
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_cb.pop()


@ttl.kernel(grid=(1, 1))
def bcast_col_multitile_kernel(inp, out):
    """Broadcast col tiles to full tiles (2x2 grid = 4 tiles)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.bcast(i, o, "col")
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_cb.pop()


@ttl.kernel(grid=(1, 1))
def bcast_scalar_multitile_kernel(inp, out):
    """Broadcast scalar tiles to full tiles (2x2 grid = 4 tiles)."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, out_cb.reserve() as o:
            result = ttl.bcast(i, o, "scalar")
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_cb.pop()


# =============================================================================
# Test Cases
# =============================================================================


class TestBcastRow:
    """Test row broadcast operation."""

    def test_bcast_row_basic(self, device):
        """Test basic row broadcast."""
        value = 3.0
        inp_torch = create_row_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        # Show input tensor (only first row is filled)
        print("\n=== Row Broadcast Test ===")
        print(f"Input tile (row 0 filled with {value}):")
        print(f"  inp[0, 0:8]  = {inp_torch[0, 0:8].tolist()}")  # First row
        print(f"  inp[1, 0:8]  = {inp_torch[1, 0:8].tolist()}")  # Second row (zeros)
        print(f"  inp[15, 0:8] = {inp_torch[15, 0:8].tolist()}")  # Middle row (zeros)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_row_kernel(inp, out)
        result = ttnn.to_torch(out)

        # Show output tensor (should be all filled)
        print(f"\nOutput tile (all rows should be {value}):")
        print(f"  out[0, 0:8]  = {result[0, 0:8].tolist()}")
        print(f"  out[1, 0:8]  = {result[1, 0:8].tolist()}")
        print(f"  out[15, 0:8] = {result[15, 0:8].tolist()}")
        print(f"  out[31, 0:8] = {result[31, 0:8].tolist()}")

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_row_negative(self, device):
        """Test row broadcast with negative value."""
        value = -2.5
        inp_torch = create_row_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_row_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_row_small(self, device):
        """Test row broadcast with small value."""
        value = 0.125
        inp_torch = create_row_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_row_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


class TestBcastCol:
    """Test column broadcast operation."""

    def test_bcast_col_basic(self, device):
        """Test basic column broadcast."""
        value = 5.0
        inp_torch = create_col_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_col_negative(self, device):
        """Test col broadcast with negative value."""
        value = -4.0
        inp_torch = create_col_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_col_large(self, device):
        """Test col broadcast with larger value."""
        value = 100.0
        inp_torch = create_col_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


class TestBcastScalar:
    """Test scalar broadcast operation."""

    def test_bcast_scalar_basic(self, device):
        """Test basic scalar broadcast."""
        value = 7.0
        inp_torch = create_scalar_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_scalar_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_scalar_one(self, device):
        """Test scalar broadcast with value 1.0."""
        value = 1.0
        inp_torch = create_scalar_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_scalar_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_scalar_negative(self, device):
        """Test scalar broadcast with negative value."""
        value = -0.5
        inp_torch = create_scalar_bcast_input(value)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
        expected = expected_bcast_result(value)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_scalar_kernel(inp, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


class TestBcastMultitile:
    """Test bcast with multi-tile CBs (2x2 tile grid = 4 tiles)."""

    def test_bcast_row_multitile(self, device):
        """Test row broadcast on 2x2 tile grid with different values per tile."""
        values = [1.0, 2.0, 3.0, 4.0]
        inp_torch = create_multitile_row_bcast_input(values)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        expected = expected_multitile_bcast_result(values)

        print("\n=== Row Broadcast Multitile Test ===")
        print("Input:")
        print(inp_torch)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_row_multitile_kernel(inp, out)
        result = ttnn.to_torch(out)

        print("Output:")
        print(result)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_col_multitile(self, device):
        """Test col broadcast on 2x2 tile grid with different values per tile."""
        values = [5.0, 6.0, 7.0, 8.0]
        inp_torch = create_multitile_col_bcast_input(values)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        expected = expected_multitile_bcast_result(values)

        print("\n=== Col Broadcast Multitile Test ===")
        print("Input:")
        print(inp_torch)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_col_multitile_kernel(inp, out)
        result = ttnn.to_torch(out)

        print("Output:")
        print(result)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_bcast_scalar_multitile(self, device):
        """Test scalar broadcast on 2x2 tile grid with different values per tile."""
        values = [0.5, 1.5, 2.5, 3.5]
        inp_torch = create_multitile_scalar_bcast_input(values)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        expected = expected_multitile_bcast_result(values)

        print("\n=== Scalar Broadcast Multitile Test ===")
        print("Input:")
        print(inp_torch)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        bcast_scalar_multitile_kernel(inp, out)
        result = ttnn.to_torch(out)

        print("Output:")
        print(result)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


class TestBcastComposition:
    """Test bcast in composition with other operations."""

    def test_mul_add_bcast_row(self, device):
        """Test (a * b) + bcast(c) pattern with row broadcast.

        a = full tile of 2.0
        b = full tile of 3.0
        c = row tile with 1.0 (will broadcast to full tile of 1.0)

        Expected: (2.0 * 3.0) + 1.0 = 7.0
        """
        a_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        b_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        c_torch = create_row_bcast_input(1.0)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Expected: (2 * 3) + 1 = 7
        expected = torch.full((32, 32), 7.0, dtype=torch.bfloat16)

        a = to_l1(a_torch, device)
        b = to_l1(b_torch, device)
        c = to_l1(c_torch, device)
        out = to_l1(out_torch, device)

        mul_add_bcast_kernel(a, b, c, out)
        result = ttnn.to_torch(out)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Fused Multitile Composition: (a * b) + bcast(c) with 2x2 tiles
# =============================================================================


@ttl.kernel(grid=(1, 1))
def mul_add_bcast_multitile_kernel(a, b, c, out):
    """Compute (a * b) + bcast(c) on 2x2 tile grid.

    Uses two-stage pattern:
    - Stage 1: bcast c to intermediate CB
    - Stage 2: compute (a * b) + c_bcast
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(2, 2), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(2, 2), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(2, 2), buffer_factor=2)
    c_bcast_cb = ttl.make_circular_buffer_like(c, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        # Stage 1: Bcast c and store to intermediate CB
        with c_cb.wait() as c_tile, c_bcast_cb.reserve() as c_out:
            c_bcast = ttl.bcast(c_tile, c_out, "row")
            c_out.store(c_bcast)

        # Stage 2: Compute (a * b) + c_bcast
        with (
            a_cb.wait() as a_tile,
            b_cb.wait() as b_tile,
            c_bcast_cb.wait() as c_bcast_tile,
            out_cb.reserve() as o,
        ):
            ab = a_tile * b_tile
            result = ab + c_bcast_tile
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as a_blk:
            tx_a = ttl.copy(a[0:2, 0:2], a_blk)
            tx_a.wait()

        with b_cb.reserve() as b_blk:
            tx_b = ttl.copy(b[0:2, 0:2], b_blk)
            tx_b.wait()

        with c_cb.reserve() as c_blk:
            tx_c = ttl.copy(c[0:2, 0:2], c_blk)
            tx_c.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


class TestBcastCompositionMultitile:
    """Test fused bcast with other ops on multi-tile CBs."""

    def test_mul_add_bcast_row_multitile(self, device):
        """Test (a * b) + bcast(c) pattern on 2x2 tile grid.

        Per-tile values:
          a: tile(i,j) = i*2 + j + 1  (1, 2, 3, 4)
          b: all tiles = 2.0
          c: row bcast with tile(i,j) = (i*2 + j + 1) * 0.5  (0.5, 1.0, 1.5, 2.0)

        Expected per tile: (a * b) + c = (a * 2) + c
          tile(0,0): (1 * 2) + 0.5 = 2.5
          tile(0,1): (2 * 2) + 1.0 = 5.0
          tile(1,0): (3 * 2) + 1.5 = 7.5
          tile(1,1): (4 * 2) + 2.0 = 10.0
        """
        # Create 64x64 tensors (2x2 tiles)
        a_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        a_torch[0:32, 0:32] = 1.0  # Tile (0,0)
        a_torch[0:32, 32:64] = 2.0  # Tile (0,1)
        a_torch[32:64, 0:32] = 3.0  # Tile (1,0)
        a_torch[32:64, 32:64] = 4.0  # Tile (1,1)

        b_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)

        c_torch = create_multitile_row_bcast_input([0.5, 1.0, 1.5, 2.0])

        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        # Expected: (a * 2) + c
        expected = torch.zeros((64, 64), dtype=torch.bfloat16)
        expected[0:32, 0:32] = 2.5  # (1 * 2) + 0.5
        expected[0:32, 32:64] = 5.0  # (2 * 2) + 1.0
        expected[32:64, 0:32] = 7.5  # (3 * 2) + 1.5
        expected[32:64, 32:64] = 10.0  # (4 * 2) + 2.0

        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        c = ttnn.from_torch(
            c_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        print("\n=== Fused Multitile Bcast Test: (a * b) + bcast(c) ===")
        print("Input a:")
        print(a_torch)
        print("Input b:")
        print(b_torch)
        print("Input c:")
        print(c_torch)

        mul_add_bcast_multitile_kernel(a, b, c, out)
        result = ttnn.to_torch(out)

        print("Output:")
        print(result)

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)

    def test_mul_add_bcast_row_multitile_constant(self, device):
        """Test (a * b) + bcast(c) with CONSTANT values across all tiles.

        All tiles use the same values:
          a: all 2.0
          b: all 3.0
          c: row bcast with 1.0 in row 0 of each tile

        Expected: (2 * 3) + 1 = 7.0 for ALL tiles
        """
        a_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)
        b_torch = torch.full((64, 64), 3.0, dtype=torch.bfloat16)

        # c: row 0 of each tile has 1.0
        c_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        c_torch[0, 0:32] = 1.0  # Tile (0,0) row 0
        c_torch[0, 32:64] = 1.0  # Tile (0,1) row 0
        c_torch[32, 0:32] = 1.0  # Tile (1,0) row 0
        c_torch[32, 32:64] = 1.0  # Tile (1,1) row 0

        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
        expected = torch.full((64, 64), 7.0, dtype=torch.bfloat16)

        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        c = ttnn.from_torch(
            c_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        print("\n=== Fused Multitile Bcast Test (Constant): (a * b) + bcast(c) ===")
        print("Input a:")
        print(a_torch)
        print("Input b:")
        print(b_torch)
        print("Input c:")
        print(c_torch)

        mul_add_bcast_multitile_kernel(a, b, c, out)
        result = ttnn.to_torch(out)

        print("Output:")
        print(result)
        print(
            f"\nExpected: all 7.0, Actual unique values: {torch.unique(result).tolist()}"
        )

        assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
