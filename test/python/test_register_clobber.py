# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for SiLU (x * sigmoid(x)) pattern.

This tests the DST register allocation fix for multi-consumer values where
a block argument is consumed by both a unary op (sigmoid) and a binary op (mul).
Without proper copy insertion, the unary op would clobber the input before
the binary op could use it.
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
