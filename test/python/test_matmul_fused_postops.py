# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression tests for elementwise ops applied to matmul results.

Prior to the fix, `o.store(sc * (a @ b) + bi)` silently dropped the scale
and bias, producing the raw matmul result. This test verifies that fused
elementwise post-ops (mul, add) on the matmul result are preserved.

Coverage includes both DFB tile scaling and Python scalar constants that lower
to `ttl.mul_unary_const`.
"""

# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, assert_pcc, to_dram

TILE = 32


@ttl.operation(grid=(1, 1))
def matmul_scale_bias_kernel(A, B, scale_tile, bias_tile, out):
    """Fused: o.store(sc * (a @ b) + bi) in a single store."""
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scale_tile, shape=(1, 1), block_count=1)
    bias_dfb = ttl.make_dataflow_buffer_like(bias_tile, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        sc = sc_dfb.wait()
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        bi = bias_dfb.wait()
        o_blk = out_dfb.reserve()
        o_blk.store(sc * (a_blk @ b_blk) + bi)
        sc.pop()
        a_blk.pop()
        b_blk.pop()
        bi.pop()
        o_blk.push()

    @ttl.datamovement()
    def dm_read():
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scale_tile[0, 0], blk)
            tx.wait()
            blk.push()
        with a_dfb.reserve() as blk:
            tx = ttl.copy(A[0, 0], blk)
            tx.wait()
            blk.push()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(B[0, 0], blk)
            tx.wait()
            blk.push()
        with bias_dfb.reserve() as blk:
            tx = ttl.copy(bias_tile[0, 0], blk)
            tx.wait()
            blk.push()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()
            blk.pop()


@ttl.operation(grid=(1, 1))
def matmul_scalar_const_scale_kernel(lhs, rhs, out):
    """Fused: o.store(0.5 * (lhs @ rhs)) using a Python scalar constant."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_blk, rhs_dfb.wait() as rhs_blk:
            with out_dfb.reserve() as out_blk:
                out_blk.store(0.5 * (lhs_blk @ rhs_blk))

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_blk:
            ttl.copy(lhs[0, 0], lhs_blk).wait()
        with rhs_dfb.reserve() as rhs_blk:
            ttl.copy(rhs[0, 0], rhs_blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0, 0]).wait()


@pytest.mark.requires_device
def test_matmul_scale_bias_fused(device):
    """Regression test for #476: scale * matmul + bias must not be dropped."""
    torch.manual_seed(42)
    A_pt = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    B_pt = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    scale_val = 0.5
    bias_val = 10.0
    sc_pt = torch.full((TILE, TILE), scale_val, dtype=torch.bfloat16)
    bi_pt = torch.full((TILE, TILE), bias_val, dtype=torch.bfloat16)

    out_tt = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)
    matmul_scale_bias_kernel(
        to_dram(A_pt, device),
        to_dram(B_pt, device),
        to_dram(sc_pt, device),
        to_dram(bi_pt, device),
        out_tt,
    )

    result = ttnn.to_torch(out_tt).reshape(TILE, TILE).float()
    golden = (
        (scale_val * (A_pt.float() @ B_pt.float()) + bias_val)
        .to(torch.bfloat16)
        .float()
    )

    assert_pcc(golden, result, threshold=0.999)


@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16", "fp32"],
)
@pytest.mark.requires_device
def test_matmul_scalar_const_scale(device, dtype):
    """Python scalar constant scale on a matmul result must not be dropped."""
    lhs_torch = torch.ones(TILE, TILE, dtype=dtype)
    rhs_torch = torch.ones(TILE, TILE, dtype=dtype)
    out_tt = to_dram(torch.zeros(TILE, TILE, dtype=dtype), device)

    matmul_scalar_const_scale_kernel(
        to_dram(lhs_torch, device),
        to_dram(rhs_torch, device),
        out_tt,
    )

    result = ttnn.to_torch(out_tt).float()
    expected = (0.5 * (lhs_torch.float() @ rhs_torch.float())).to(dtype).float()
    assert_allclose(result, expected, rtol=1e-2, atol=1e-2)
