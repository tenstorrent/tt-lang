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
SCALAR_CONST_SCALE = 0.5


def _make_distinct_tile_constants(tile_rows, tile_cols, dtype, row_weight, col_weight):
    matrix = torch.empty(tile_rows * TILE, tile_cols * TILE, dtype=torch.float32)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            value = 1.0 + row_weight * tile_row + col_weight * tile_col
            row_start = tile_row * TILE
            col_start = tile_col * TILE
            matrix[
                row_start : row_start + TILE,
                col_start : col_start + TILE,
            ] = value
    return matrix.to(dtype)


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
    lhs_rows = lhs.shape[0] // TILE
    shared_tiles = lhs.shape[1] // TILE
    rhs_cols = rhs.shape[1] // TILE
    lhs_dfb = ttl.make_dataflow_buffer_like(
        lhs, shape=(lhs_rows, shared_tiles), block_count=2
    )
    rhs_dfb = ttl.make_dataflow_buffer_like(
        rhs, shape=(shared_tiles, rhs_cols), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(lhs_rows, rhs_cols), block_count=2
    )

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as lhs_blk, rhs_dfb.wait() as rhs_blk:
            with out_dfb.reserve() as out_blk:
                out_blk.store(SCALAR_CONST_SCALE * (lhs_blk @ rhs_blk))

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_blk:
            ttl.copy(lhs[0:lhs_rows, 0:shared_tiles], lhs_blk).wait()
        with rhs_dfb.reserve() as rhs_blk:
            ttl.copy(rhs[0:shared_tiles, 0:rhs_cols], rhs_blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:lhs_rows, 0:rhs_cols]).wait()


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
    "block_shape,dtype",
    [
        pytest.param((1, 1, 1), torch.bfloat16, id="1x1x1_bf16"),
        pytest.param((2, 2, 2), torch.bfloat16, id="2x2x2_bf16"),
        pytest.param((1, 2, 4), torch.bfloat16, id="1x2x4_bf16"),
        pytest.param((1, 1, 1), torch.float32, id="1x1x1_fp32"),
        pytest.param((2, 2, 2), torch.float32, id="2x2x2_fp32"),
        pytest.param((1, 2, 4), torch.float32, id="1x2x4_fp32"),
    ],
)
@pytest.mark.requires_device
def test_matmul_scalar_const_scale(device, block_shape, dtype):
    """Python scalar constant scale on a matmul result must not be dropped."""
    lhs_rows, shared_tiles, rhs_cols = block_shape
    lhs_torch = _make_distinct_tile_constants(
        lhs_rows, shared_tiles, dtype, row_weight=0.25, col_weight=0.125
    )
    rhs_torch = _make_distinct_tile_constants(
        shared_tiles, rhs_cols, dtype, row_weight=0.375, col_weight=-0.125
    )
    out_torch = torch.zeros(lhs_rows * TILE, rhs_cols * TILE, dtype=dtype)
    out_tt = to_dram(out_torch, device)

    matmul_scalar_const_scale_kernel(
        to_dram(lhs_torch, device),
        to_dram(rhs_torch, device),
        out_tt,
    )

    result = ttnn.to_torch(out_tt).float()
    expected = (
        (SCALAR_CONST_SCALE * (lhs_torch.float() @ rhs_torch.float())).to(dtype).float()
    )
    tolerance = {"rtol": 5e-3, "atol": 1e-2}
    if dtype == torch.bfloat16:
        tolerance = {"rtol": 1e-2, "atol": 1.0}
    assert_allclose(result, expected, **tolerance)
