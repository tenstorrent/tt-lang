# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for axby bug: a*x + b*y with 4 separate circular buffers.

This test validates correct register allocation for binary operations when
multiple intermediate results need to be tracked. The bug manifested as
incorrect source register operands in the ConvertTTLToTTKernel pass.

Expected computation:
  term1 = a * x
  term2 = b * y
  result = term1 + term2

With test values a=2, x=3, b=4, y=5:
  term1 = 6
  term2 = 20
  result = 26

The bug caused incorrect register operands for the second multiply,
computing (a*x) * b instead of b * y, resulting in 30 instead of 26.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttnn
from ttl import ttl
from test_helpers import assert_allclose

# Skip all tests if ttnn not available
pytestmark = pytest.mark.requires_ttnn


@ttl.kernel(grid=(1, 1))
def axby_fused_kernel(a, x, b, y, out):
    """
    Fused kernel: out = a*x + b*y with 4 separate CBs.

    This tests proper DST register allocation across multiple
    binary operations with distinct operands.
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    y_cb = ttl.make_circular_buffer_like(y, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_cb.wait() as av, x_cb.wait() as xv, b_cb.wait() as bv, y_cb.wait() as yv:
            with out_cb.reserve() as o:
                # Each operand from a different CB
                term1 = av * xv
                term2 = bv * yv
                result = term1 + term2
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk)
            tx.wait()
        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()
        with y_cb.reserve() as blk:
            tx = ttl.copy(y[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

    return ttl.Program(compute, dm_read, dm_write)(a, x, b, y, out)


def test_axby_fused_multiply_add():
    """Test a*x + b*y pattern with 4 separate circular buffers."""
    device = ttnn.open_device(device_id=0)

    try:
        # Test values: a=2, x=3, b=4, y=5
        # Expected: 2*3 + 4*5 = 6 + 20 = 26
        a_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        x_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        b_torch = torch.full((32, 32), 4.0, dtype=torch.bfloat16)
        y_torch = torch.full((32, 32), 5.0, dtype=torch.bfloat16)
        out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

        a_t = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_t = ttnn.from_torch(
            x_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b_t = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        y_t = ttnn.from_torch(
            y_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_t = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Expected: a*x + b*y
        expected = a_torch * x_torch + b_torch * y_torch

        # Run kernel
        axby_fused_kernel(a_t, x_t, b_t, y_t, out_t)
        result = ttnn.to_torch(out_t)

        # Verify correctness
        assert_allclose(result, expected, rtol=0.01, atol=0.1)

    finally:
        ttnn.close_device(device)
