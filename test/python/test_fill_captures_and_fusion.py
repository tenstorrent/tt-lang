# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for fill with captured float variables and fill+elementwise fusion.

Tests two patterns:
1. Captured float variables from enclosing scope used with ttl.math.fill
2. Fill fused with elementwise ops in a single ttl.compute region
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import math

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


# =============================================================================
# Pattern 1: Captured float variables
# =============================================================================


def _make_fill_captured_kernel(fill_value):
    """Build a fill kernel that captures a Python float from the enclosing scope."""
    c_val = float(fill_value)

    @ttl.kernel(grid=(1, 1))
    def fill_captured_kernel(inp, out):
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute_fn():
            with inp_cb.wait() as x, out_cb.reserve() as o:
                o.store(ttl.math.fill(o, c_val))

        @ttl.datamovement()
        def dm_read():
            with inp_cb.reserve() as inp_blk:
                tx = ttl.copy(inp[0, 0], inp_blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_cb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    return fill_captured_kernel


# =============================================================================
# Pattern 2: Fill fused with elementwise ops
# =============================================================================


def _make_fill_exp_fused_kernel():
    """Fill with 1.0 then exp -> should produce e everywhere."""

    @ttl.kernel(grid=(1, 1))
    def fill_exp_kernel(inp, out):
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute_fn():
            with inp_cb.wait() as x, out_cb.reserve() as o:
                o.store(ttl.math.exp(ttl.math.fill(o, 1.0)))

        @ttl.datamovement()
        def dm_read():
            with inp_cb.reserve() as inp_blk:
                tx = ttl.copy(inp[0, 0], inp_blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_cb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    return fill_exp_kernel


def _make_fill_mul_fused_kernel(scale_value):
    """Fill with a captured constant then multiply with input.

    Computes: input * fill(scale_value)
    This tests fill as one operand in a binary fusion chain.
    """
    c_scale = float(scale_value)

    @ttl.kernel(grid=(1, 1))
    def fill_mul_kernel(inp, out):
        inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute_fn():
            with inp_cb.wait() as x, out_cb.reserve() as o:
                o.store(x * ttl.math.fill(o, c_scale))

        @ttl.datamovement()
        def dm_read():
            with inp_cb.reserve() as inp_blk:
                tx = ttl.copy(inp[0, 0], inp_blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_cb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    return fill_mul_kernel


# =============================================================================
# Tests
# =============================================================================


def test_fill_captured_float(device):
    """Test fill with a captured Python float variable."""
    kernel = _make_fill_captured_kernel(42.0)
    inp_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.full((32, 32), 42.0, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fill_captured_computed_float(device):
    """Test fill with a captured float computed from other Python values."""
    box_length = 10.0
    c_inv_box = 1.0 / box_length

    kernel = _make_fill_captured_kernel(c_inv_box)
    inp_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.full((32, 32), c_inv_box, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fill_exp_fusion(device):
    """Test fill(1.0) fused with exp -> produces e."""
    kernel = _make_fill_exp_fused_kernel()
    inp_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.full((32, 32), math.e, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fill_mul_fusion(device):
    """Test input * fill(scale) fusion: scales input by a constant."""
    kernel = _make_fill_mul_fused_kernel(3.0)
    inp_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.full((32, 32), 6.0, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    out = to_l1(out_torch, device)

    kernel(inp, out)
    result = ttnn.to_torch(out)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
