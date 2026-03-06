# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Reproducer tests for known issues.

Issue 365: Non-square matmul output corrupts CB memory with small buffer_factor
Issue 363: Multicore fused matmul miscompile when K >= 5 tiles (HW only)
Issue 364: Reduce ops cause CB memory aliasing with shared buffer_factor
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1

import ttl


# =============================================================================
# Issue 365: Non-square matmul output with small buffer_factor
# https://github.com/tenstorrent/tt-lang/issues/365
#
# Non-square matmul A[1,4] @ B[4,4] -> C[1,4] followed by elementwise
# accumulate. With small buffer_factor, the matmul codegen wrote beyond the
# reserved CB tiles. Fixed by correct loop bound generation.
# =============================================================================


@ttl.kernel(grid=(1, 1))
def issue365_matmul_nonsquare_kernel(a, b, c):
    """Non-square matmul: C[1,4] = A[1,4] @ B[4,4]."""
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 4), buffer_factor=2)
    b_cb = ttl.make_dataflow_buffer_like(b, shape=(4, 4), buffer_factor=2)
    c_cb = ttl.make_dataflow_buffer_like(c, shape=(1, 4), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as av, b_cb.wait() as bv, c_cb.reserve() as cv:
            cv.store(ttl.math.matmul(av, bv))

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0:1, 0:4], blk)
            tx.wait()

        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0:4, 0:4], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_cb.wait() as blk:
            tx = ttl.copy(blk, c[0:1, 0:4])
            tx.wait()


def test_issue365_nonsquare_matmul(device):
    """Reproducer for issue 365: non-square matmul output shape.

    A[32, 128] @ B[128, 128] -> C[32, 128] (tile shape [1,4] @ [4,4] = [1,4]).
    With all ones: C[i,j] = sum_k(A[i,k] * B[k,j]) = 128 * 1 = 128
    """
    a_torch = torch.ones((32, 128), dtype=torch.bfloat16)
    b_torch = torch.ones((128, 128), dtype=torch.bfloat16)
    c_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

    expected = torch.full((32, 128), 128.0, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    c = to_l1(c_torch, device)

    issue365_matmul_nonsquare_kernel(a, b, c)
    result = ttnn.to_torch(c)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1.0)


# =============================================================================
# Issue 363: Multicore fused matmul miscompile when K >= 5
# https://github.com/tenstorrent/tt-lang/issues/363
#
# Fused matmul with K >= 5 tiles on multicore grid produces wrong results
# on hardware. Simulator produces correct results.
# =============================================================================


@ttl.kernel(grid=(1, 1))
def issue363_k5_matmul_kernel(a, b, c):
    """Matmul with K=5: A[1,5] @ B[5,1] -> C[1,1]."""
    a_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 5), buffer_factor=2)
    b_cb = ttl.make_dataflow_buffer_like(b, shape=(5, 1), buffer_factor=2)
    c_cb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as av, b_cb.wait() as bv, c_cb.reserve() as cv:
            cv.store(ttl.math.matmul(av, bv))

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as blk:
            tx = ttl.copy(a[0:1, 0:5], blk)
            tx.wait()

        with b_cb.reserve() as blk:
            tx = ttl.copy(b[0:5, 0:1], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_cb.wait() as blk:
            tx = ttl.copy(blk, c[0, 0])
            tx.wait()


def test_issue363_k5_matmul(device):
    """Reproducer for issue 363: matmul with K=5 tiles.

    A[32, 160] @ B[160, 32] -> C[32, 32] (tile shape [1,5] @ [5,1] = [1,1]).
    With all ones: C[i,j] = sum_k(1*1) = 160
    This passes on simulator but fails on hardware when K >= 5.
    """
    a_torch = torch.ones((32, 160), dtype=torch.bfloat16)
    b_torch = torch.ones((160, 32), dtype=torch.bfloat16)
    c_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    expected = torch.full((32, 32), 160.0, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    c = to_l1(c_torch, device)

    issue363_k5_matmul_kernel(a, b, c)
    result = ttnn.to_torch(c)

    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=2.0)


# =============================================================================
# Issue 364: Reduce CB aliasing with shared buffer_factor
# https://github.com/tenstorrent/tt-lang/issues/364
#
# Two reduce operations writing to different CBs with same buffer_factor
# causes silent aliasing: reading from one CB returns the other's result.
# =============================================================================


@ttl.kernel(grid=(1, 1))
def issue364_reduce_basic_kernel(inp, scaler, out):
    """Basic single reduce to verify reduce works at all."""
    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_cb = ttl.make_dataflow_buffer_like(
        scaler, shape=(1, 1), buffer_factor=2
    )
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_cb.wait() as i, scaler_cb.wait() as s, out_cb.reserve() as o:
            o.store(ttl.math.reduce_sum(i, s, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def test_issue364_basic_reduce(device):
    """Reproducer for issue 364: basic single reduce.

    Input: 32x32 tile of all 3.0, scaler: all 1.0
    Expected: scalar sum = 3.0 * 32 * 32 = 3072.0 (broadcast across output tile)

    This tests that a single reduce works correctly before testing aliasing.
    """
    inp_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    scaler_torch = torch.ones((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    a = to_l1(inp_torch, device)
    s = to_l1(scaler_torch, device)
    o = to_l1(out_torch, device)

    issue364_reduce_basic_kernel(a, s, o)
    result = ttnn.to_torch(o)

    # The reduce_sum with scaler=1.0 should give sum of all elements
    # With 32x32 tiles, sum of 3.0 = 3072.0
    # Result is broadcast to fill the entire output tile
    expected_val = 3.0 * 32 * 32
    print(f"\nIssue 364 basic reduce: expected ~{expected_val}, got {result[0, 0].item()}")

    assert_allclose(
        result[0, 0].float(),
        torch.tensor(expected_val, dtype=torch.float32),
        rtol=0.1,
        atol=10.0,
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
