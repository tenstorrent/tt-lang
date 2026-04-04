# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test for fill operation: fill tiles with a constant value.

Tests:
- Multi-tile (2x2) fill with negative constant (-3.0)
- Fill fused with elementwise add (fill(1.0) + input)
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


@ttl.operation(grid=(1, 1))
def fill_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as _in, out_dfb.reserve() as o:
            o.store(ttl.math.fill(o, -3.0))

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        ttl.copy(inp[0:2, 0:2], inp_blk).wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        ttl.copy(out_blk, out[0:2, 0:2]).wait()
        out_blk.pop()


@ttl.operation(grid=(1, 1))
def fill_add_kernel(inp, out):
    """Fill with 1.0 then add to input: out = inp + 1.0."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            filled = ttl.math.fill(o, 1.0)
            o.store(x + filled)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_fill_negative_constant(device):
    """Test multi-tile fill with negative constant value."""
    inp = to_l1(torch.zeros((64, 64), dtype=torch.bfloat16), device)
    out = to_l1(torch.zeros((64, 64), dtype=torch.bfloat16), device)

    fill_kernel(inp, out)
    result = ttnn.to_torch(out)

    expected = torch.full((64, 64), -3.0, dtype=torch.bfloat16)
    assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


def test_fill_fused_with_add(device):
    """Test fill(1.0) fused with elementwise add: out = inp + 1.0."""
    inp_torch = torch.full((32, 32), 5.0, dtype=torch.bfloat16)
    inp = to_l1(inp_torch, device)
    out = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    fill_add_kernel(inp, out)
    result = ttnn.to_torch(out)

    expected = inp_torch + 1.0
    assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
