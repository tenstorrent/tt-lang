# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for compiler-allocated scratch dataflow buffers.

The ttl-insert-intermediate-dfbs pass automatically creates DFBs when a
fused expression chain feeds into an op that requires DFB-attached inputs
(reduce, bcast, matmul, transpose). The user does not declare intermediate
DFBs; the compiler inserts them at split points.
"""

import pytest
import torch
import ttnn
import ttl
from ttlang_test_utils import assert_allclose, to_l1

pytestmark = pytest.mark.requires_device


# --- add -> reduce_sum (elementwise feeds CB-input op) ---


@ttl.operation(grid=(1, 1))
def add_then_reduce_kernel(a, b, scaler, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, scaler_dfb.wait() as sv:
            # add result is not DFB-attached; compiler inserts scratch DFB.
            added = ttl.add(av, bv)
            with out_dfb.reserve() as o:
                o.store(ttl.math.reduce_sum(added, sv, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b[0, 0], blk).wait()
        with scaler_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_add_then_reduce(device):
    """Elementwise add feeds reduce_sum; compiler inserts scratch DFB."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    expected = (a_torch.float() + b_torch.float()).sum()

    add_then_reduce_kernel(a, b, scaler, out)
    result = ttnn.to_torch(out).float()

    # Scalar reduce: result is at [0,0].
    assert_allclose(result[0, 0], expected, rtol=0.01, atol=0.5)


# --- matmul -> reduce_sum (matmul result feeds CB-input op) ---


@ttl.operation(grid=(1, 1))
def matmul_then_reduce_kernel(a, b, scaler, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with (
            a_dfb.wait() as av,
            b_dfb.wait() as bv,
            scaler_dfb.wait() as sv,
        ):
            # matmul result is not DFB-attached; compiler inserts scratch DFB.
            product = av @ bv  # matmul via __matmul__
            with out_dfb.reserve() as o:
                o.store(ttl.math.reduce_sum(product, sv, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b[0, 0], blk).wait()
        with scaler_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_matmul_then_reduce(device):
    """Matmul result feeds reduce_sum; compiler inserts scratch DFB."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    expected = (a_torch.float() @ b_torch.float()).sum()

    matmul_then_reduce_kernel(a, b, scaler, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result[0, 0], expected, rtol=0.01, atol=1.0)
