# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for DST register allocation with multi-consumer block arguments.

Tests two patterns:
1. SiLU (x * sigmoid(x)): block arg consumed by unary and binary ops
2. Unary+Binary: block arg consumed by one unary op and two binary ops

Without proper copy insertion, unary ops would clobber inputs before
other consumers could use them.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttnn
from test_helpers import assert_allclose
from ttl import ttl

pytestmark = pytest.mark.requires_ttnn


@ttl.kernel(grid=(1, 1))
def silu_kernel(x, out):
    """SiLU: x * sigmoid(x) - tests multi-consumer DST allocation."""
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv:
            with out_cb.reserve() as o:
                sig = ttl.math.sigmoid(xv)
                result = xv * sig
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as blk:
            tx = ttl.copy(x[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

    return ttl.Program(compute, dm_read, dm_write)(x, out)


@pytest.fixture(scope="module")
def device():
    """Open device once per module."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@ttl.kernel(grid=(1, 1))
def unary_binary_kernel(x, y, out):
    """Tests block arg with one unary and two binary consumers."""
    x_cb = ttl.make_circular_buffer_like(x, shape=(1, 1), buffer_factor=2)
    y_cb = ttl.make_circular_buffer_like(y, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with x_cb.wait() as xv, y_cb.wait() as yv:
            with out_cb.reserve() as o:
                # x is used by: abs (unary), add (binary), mul (binary)
                abs_x = ttl.math.abs(xv)
                add_result = xv + yv
                mul_result = xv * yv
                # Combine all results
                result = abs_x + add_result + mul_result
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with x_cb.reserve() as x_blk, y_cb.reserve() as y_blk:
            tx = ttl.copy(x[0, 0], x_blk)
            ty = ttl.copy(y[0, 0], y_blk)
            tx.wait()
            ty.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()

    return ttl.Program(compute, dm_read, dm_write)(x, y, out)


def test_silu(device):
    """Test SiLU activation: x * sigmoid(x)."""
    x_torch = torch.tensor(
        [[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] + [1.0] * 26] * 32, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    expected = x_torch * torch.sigmoid(x_torch.float()).to(torch.bfloat16)

    x_tensor = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tensor = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    silu_kernel(x_tensor, out_tensor)

    result = ttnn.to_torch(out_tensor)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_unary_binary_consumers(device):
    """Test block arg used by one unary op and two binary ops."""
    x_torch = torch.tensor(
        [[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0] + [0.5] * 26] * 32, dtype=torch.bfloat16
    )
    y_torch = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [2.0] * 26] * 32, dtype=torch.bfloat16
    )
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    # Expected: abs(x) + (x + y) + (x * y)
    expected = (
        torch.abs(x_torch.float())
        + (x_torch.float() + y_torch.float())
        + (x_torch.float() * y_torch.float())
    ).to(torch.bfloat16)

    x_tensor = ttnn.from_torch(
        x_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_tensor = ttnn.from_torch(
        y_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tensor = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    unary_binary_kernel(x_tensor, y_tensor, out_tensor)

    result = ttnn.to_torch(out_tensor)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
