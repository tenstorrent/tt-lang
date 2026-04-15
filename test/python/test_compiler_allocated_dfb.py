# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for compiler-allocated intermediate dataflow buffers.

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


# --- add -> reduce_sum (elementwise feeds reduce) ---


@ttl.operation(grid=(1, 1))
def add_then_reduce_kernel(a, b, scaler, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, scaler_dfb.wait() as sv:
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
    """Elementwise add feeds reduce_sum; compiler inserts intermediate DFB."""
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

    assert_allclose(result[0, 0], expected, rtol=0.01, atol=0.5)


# --- matmul -> reduce_sum (matmul result feeds reduce) ---


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
            product = av @ bv
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
    """Matmul result feeds reduce_sum; compiler inserts intermediate DFB."""
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


@ttl.operation(grid=(1, 1))
def add_then_bcast_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            # add produces a non-CB-attached value; broadcast needs
            # CB-attached input. Compiler inserts intermediate DFB.
            added = ttl.add(av, bv)
            result = ttl.math.broadcast(added, o, dims=[0, 1])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_add_then_bcast(device):
    """Elementwise add feeds broadcast; compiler inserts intermediate DFB."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    out = to_l1(out_torch, device)

    added = a_torch.float() + b_torch.float()
    # Scalar broadcast copies element [0,0] to all 32x32 positions.
    expected = torch.full((32, 32), added[0, 0].item())

    add_then_bcast_kernel(a, b, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result, expected, rtol=0.01, atol=0.5)


@ttl.operation(grid=(1, 1))
def add_then_transpose_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            added = ttl.add(av, bv)
            o.store(ttl.math.transpose(added))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a[0, 0], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_add_then_transpose(device):
    """Elementwise add feeds transpose; compiler inserts intermediate DFB."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    out = to_l1(out_torch, device)

    expected = (a_torch.float() + b_torch.float()).T

    add_then_transpose_kernel(a, b, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result, expected, rtol=1e-2, atol=1e-2)


@ttl.operation(grid=(1, 1))
def reduce_then_bcast_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, scaler_dfb.wait() as s, out_dfb.reserve() as o:
            # reduce_sum produces non-CB-attached result; broadcast needs
            # CB-attached input. Compiler inserts intermediate DFB between them.
            reduced = ttl.math.reduce_sum(x, s, dims=[0, 1])
            o.store(ttl.math.broadcast(reduced, o, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()
        with scaler_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_reduce_then_bcast(device):
    """Reduce output feeds broadcast; compiler inserts intermediate DFB."""
    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    # Scalar reduce then scalar broadcast: every output element = sum(input).
    expected = torch.full((32, 32), inp_torch.float().sum().item())

    reduce_then_bcast_kernel(inp, scaler, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result, expected, rtol=0.01, atol=0.5)


# mixed consumers: same value feeds reduce (needs DFB) and mul (fuses)
@ttl.operation(grid=(1, 1))
def mixed_consumers_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, scaler_dfb.wait() as s, out_dfb.reserve() as o:
            ex = ttl.exp(x)
            # ex feeds reduce_sum (needs DFB) and also mul (should fuse).
            sm = ttl.math.reduce_sum(ex, s, dims=[0, 1])
            inv = ttl.recip(ttl.math.broadcast(sm, ex, dims=[0, 1]))
            o.store(ttl.mul(ex, inv))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()
        with scaler_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_mixed_consumers(device):
    """Same value feeds reduce (DFB) and mul (fuses); verifies fusion is preserved."""
    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    # exp(x) / sum(exp(x)) = softmax-like normalization.
    ex = inp_torch.float().exp()
    expected = ex / ex.sum()

    mixed_consumers_kernel(inp, scaler, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result, expected, rtol=0.05, atol=1e-3)
