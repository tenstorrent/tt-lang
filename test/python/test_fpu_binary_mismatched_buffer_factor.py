# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for issue #438: FPU binary mul with mismatched buffer_factors.

When input DFBs have different buffer_factors (e.g., 1 vs 2), the FPU binary
lowering incorrectly compared total CB tile counts (including buffer_factor)
instead of per-block tile counts. This caused tile_mul to fail legalization.

The pattern: an outer multi-wait scope with buffer_factor=1 DFBs and an inner
scope with buffer_factor=2 DFBs. Multiplying across scopes produces operands
from CBs with different total tile counts but identical per-block shapes.

Expected computation: out = cv * bv = 5.0 * 3.0 = 15.0
"""


import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram


@ttl.operation(grid=(1, 1))
def mul_mismatched_bf(a, b, c, out):
    """Multiply cv * bv where c_dfb has buffer_factor=2, b_dfb has buffer_factor=1."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=1)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=1)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with c_dfb.wait() as cv, out_dfb.reserve() as o:
                o.store(cv * bv)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(c[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_mul_mismatched_buffer_factor(device):
    """Issue #438: tile_mul must work when operand DFBs have different buffer_factors."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    c_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    expected = c_torch * b_torch

    out_t = to_dram(out_torch, device)
    mul_mismatched_bf(
        to_dram(a_torch, device),
        to_dram(b_torch, device),
        to_dram(c_torch, device),
        out_t,
    )
    result = ttnn.to_torch(out_t)

    assert_allclose(result, expected, rtol=0.01, atol=0.1)
