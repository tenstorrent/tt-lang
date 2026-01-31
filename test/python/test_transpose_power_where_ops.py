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


# =============================================================================
# Transpose Tests
# =============================================================================


def test_transpose_single_tile(device):
    """Test single-tile transpose (1x1 CB shape)."""
    import ttl

    @ttl.kernel(grid=(1, 1))
    def transpose_1x1_kernel(inp, out):
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

    # Create input with distinct values to verify transpose
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
    import ttl

    @ttl.kernel(grid=(1, 1))
    def transpose_2x2_kernel(inp, out):
        # Input is [2, 2] tiles, output is [2, 2] tiles (transposed)
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

    # Create 64x64 input (2x2 tiles of 32x32)
    inp_torch = torch.arange(64 * 64, dtype=torch.bfloat16).reshape(64, 64)
    out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)
    expected = inp_torch.T

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    transpose_2x2_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


# =============================================================================
# Power Tests
# =============================================================================


def test_power_square(device):
    """Test power operation with exponent=2 (square)."""
    import ttl

    @ttl.kernel(grid=(1, 1))
    def power_kernel(inp, out):
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

    inp_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.pow(inp_torch, 2)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    power_kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_power_cube(device):
    """Test power operation with exponent=3 (cube)."""
    import ttl

    @ttl.kernel(grid=(1, 1))
    def power_cube_kernel(inp, out):
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
    import ttl

    @ttl.kernel(grid=(1, 1))
    def where_kernel(cond, true_val, false_val, out):
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

    # Condition: half True (1.0), half False (0.0)
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
