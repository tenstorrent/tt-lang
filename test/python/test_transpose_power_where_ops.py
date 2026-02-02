# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for transpose, power, and where operations.

Tests these ops against NumPy/PyTorch equivalents with L1 memory configuration.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import numpy as np

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


# =============================================================================
# Transpose Kernels
# =============================================================================


@ttl.kernel(grid=(1, 1))
def transpose_1x1_kernel(inp, out):
    """Single tile transpose: transpose a single 32x32 tile."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = ttl.transpose(x, o)
        o.store(result)
        inp_cb.pop()
        out_cb.push()

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
def transpose_2x2_kernel(inp, out):
    """Multi-tile transpose: 2x2 tiles -> 2x2 tiles (square case).

    Input [A B; C D] becomes [A^T C^T; B^T D^T].
    Both tile rearrangement and per-tile transpose happen.
    """
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = ttl.transpose(x, o)
        o.store(result)
        inp_cb.pop()
        out_cb.push()

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
def transpose_4x2_kernel(inp, out):
    """Non-square transpose: 4x2 tiles -> 2x4 tiles.

    Input 128x64 (4 rows, 2 cols of tiles) becomes 64x128 (2 rows, 4 cols).
    Tests that tile rearrangement works for non-square cases.
    """
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(4, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 4), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = ttl.transpose(x, o)
        o.store(result)
        inp_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        tx = ttl.copy(inp[0:4, 0:2], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:4])
        tx.wait()
        out_cb.pop()


# =============================================================================
# Power Kernels
# =============================================================================


@ttl.kernel(grid=(1, 1))
def power_square_kernel(inp, out):
    """Power operation with exponent=2."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = ttl.power(x, 2)
        o.store(result)
        inp_cb.pop()
        out_cb.push()

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
def power_cube_kernel(inp, out):
    """Power operation with exponent=3."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        result = ttl.power(x, 3)
        o.store(result)
        inp_cb.pop()
        out_cb.push()

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
# Where Kernels
# =============================================================================


@ttl.kernel(grid=(1, 1))
def where_kernel(cond, true_val, false_val, out):
    """Ternary where: out = cond ? true_val : false_val."""
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


# =============================================================================
# Transpose Tests
# =============================================================================


def test_transpose_single_tile(device):
    """Test single-tile transpose (1x1 CB shape)."""
    inp_torch = torch.arange(32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = inp_torch.T

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    transpose_1x1_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_transpose_multi_tile(device):
    """Test multi-tile transpose (2x2 CB shape).

    A 2x2 block of tiles representing:
    [A B]
    [C D]

    Should become:
    [A^T C^T]
    [B^T D^T]

    where ^T means each tile is transposed.
    """
    inp_torch = torch.arange(64 * 64, dtype=torch.bfloat16).reshape(64, 64)
    out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
    expected = inp_torch.T

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    transpose_2x2_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_transpose_non_square(device):
    """Test non-square transpose (4x2 -> 2x4 CB shape).

    Input shape: 128x64 (4 rows x 2 cols of 32x32 tiles)
    Output shape: 64x128 (2 rows x 4 cols of 32x32 tiles)

    This tests that the transpose correctly handles:
    1. Tile position rearrangement (tiles move to transposed positions)
    2. Per-tile transpose (each 32x32 tile has rows/cols swapped)
    """
    inp_torch = torch.arange(128 * 64, dtype=torch.bfloat16).reshape(128, 64)
    out_torch = torch.zeros((64, 128), dtype=torch.bfloat16)
    expected = inp_torch.T

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    transpose_4x2_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Power Tests
# =============================================================================


def test_power_square(device):
    """Test power operation with exponent=2 (square)."""
    inp_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.pow(inp_torch, 2)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    power_square_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_power_cube(device):
    """Test power operation with exponent=3 (cube)."""
    inp_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.pow(inp_torch, 3)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    power_cube_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Where Tests
# =============================================================================


def test_where_basic(device):
    """Test where operation with basic condition."""
    cond_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    cond_torch[:, :16] = 1.0  # Left half is True

    true_torch = torch.full((32, 32), 10.0, dtype=torch.bfloat16)
    false_torch = torch.full((32, 32), -10.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    expected = torch.where(cond_torch > 0, true_torch, false_torch)

    cond = to_l1(cond_torch, device)
    true_val = to_l1(true_torch, device)
    false_val = to_l1(false_torch, device)
    out = to_l1(out_torch, device)

    where_kernel(cond, true_val, false_val, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
