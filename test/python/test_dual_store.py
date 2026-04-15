# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Dual-store matmul: same result stored to two output DFBs.

Verifies that convert-ttl-to-compute correctly relocates cb_push ops
for multi-output compute ops, and that TTLLowerMatmulBlock emits
tile_store ops for all output views.
"""

import pytest
import torch
import ttnn
import ttl
from ttlang_test_utils import assert_pcc, to_l1

pytestmark = pytest.mark.requires_device


@ttl.operation(grid=(1, 1))
def dual_store_kernel(a, b, out1, out2):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    o1_dfb = ttl.make_dataflow_buffer_like(out1, shape=(1, 1), block_count=2)
    o2_dfb = ttl.make_dataflow_buffer_like(out2, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            result = av @ bv
            with o1_dfb.reserve() as o1:
                o1.store(result)
            with o2_dfb.reserve() as o2:
                o2.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with o1_dfb.wait() as blk:
            ttl.copy(blk, out1[0, 0]).wait()
        with o2_dfb.wait() as blk:
            ttl.copy(blk, out2[0, 0]).wait()


def test_dual_store_matmul(device):
    """Matmul result stored to two DFBs; both must receive correct data."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    o1_torch = torch.zeros(32, 32, dtype=torch.bfloat16)
    o2_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    o1 = to_l1(o1_torch, device)
    o2 = to_l1(o2_torch, device)

    dual_store_kernel(a, b, o1, o2)

    expected = a_torch.float() @ b_torch.float()
    assert_pcc(expected, ttnn.to_torch(o1).float(), threshold=0.999)
    assert_pcc(expected, ttnn.to_torch(o2).float(), threshold=0.999)


# --- Different results to different output CBs ---


@ttl.operation(grid=(1, 1))
def two_results_kernel(a, b, out_sum, out_prod):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(out_sum, shape=(1, 1), block_count=2)
    prod_dfb = ttl.make_dataflow_buffer_like(out_prod, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with sum_dfb.reserve() as o_sum:
                o_sum.store(ttl.add(av, bv))
            with prod_dfb.reserve() as o_prod:
                o_prod.store(ttl.mul(av, bv))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with sum_dfb.wait() as blk:
            ttl.copy(blk, out_sum[0, 0]).wait()
        with prod_dfb.wait() as blk:
            ttl.copy(blk, out_prod[0, 0]).wait()


def test_two_different_results(device):
    """Two different values (add and mul) stored to separate output DFBs."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    o_sum = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    o_prod = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    two_results_kernel(a, b, o_sum, o_prod)

    assert_pcc(
        a_torch.float() + b_torch.float(), ttnn.to_torch(o_sum).float(), threshold=0.999
    )
    assert_pcc(
        a_torch.float() * b_torch.float(),
        ttnn.to_torch(o_prod).float(),
        threshold=0.999,
    )


# --- Store to output + thread-local, then read back from local ---


@ttl.operation(grid=(1, 1))
def store_and_reuse_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
    local_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            result = av @ bv
            # Store matmul result to both output and thread-local DFB.
            with out_dfb.reserve() as o:
                o.store(result)
            with local_dfb.reserve() as loc:
                loc.store(result)

        # Read back from local, apply exp, overwrite output.
        with local_dfb.wait() as loc_val:
            with out_dfb.reserve() as o2:
                o2.store(ttl.exp(loc_val))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        # First push is the matmul result; second push overwrites with exp.
        # DM write sees the last value pushed.
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_store_and_reuse(device):
    """Matmul to output + local DFB, then exp(local) overwrites output."""
    # Scale inputs so matmul results stay in exp's valid range.
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

    store_and_reuse_kernel(a, b, out)

    # The second store (exp of matmul) is what DM write reads.
    expected = (a_torch.float() @ b_torch.float()).exp()
    assert_pcc(expected, ttnn.to_torch(out).float(), threshold=0.99)
