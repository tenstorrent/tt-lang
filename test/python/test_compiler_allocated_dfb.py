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


# --- reduce with non-CB-attached scaler (operand 1 materialization) ---


@ttl.operation(grid=(1, 1))
def reduce_computed_scaler_kernel(inp, ones, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    ones_dfb = ttl.make_dataflow_buffer_like(ones, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x, ones_dfb.wait() as s, out_dfb.reserve() as o:
            # Compute scaler from elementwise — not CB-attached.
            # reduce_sum needs both operands CB-attached, so the compiler
            # materializes the scaler (operand 1) to an intermediate DFB.
            half_scaler = ttl.mul(s, s)
            o.store(ttl.math.reduce_sum(x, half_scaler, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()
        with ones_dfb.reserve() as blk:
            ttl.copy(ones[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_reduce_computed_scaler(device):
    """Reduce scaler from elementwise; compiler materializes operand 1."""
    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    ones_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    inp = to_l1(inp_torch, device)
    ones = to_l1(ones_torch, device)
    out = to_l1(out_torch, device)

    # reduce_sum with scaler = ones * ones = ones. Result = sum(inp).
    expected = inp_torch.float().sum()

    reduce_computed_scaler_kernel(inp, ones, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result[0, 0], expected, rtol=0.01, atol=0.5)


# --- matmul with both operands non-CB-attached ---


@ttl.operation(grid=(1, 1))
def matmul_both_intermediates_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, out_dfb.reserve() as o:
            # Both matmul operands come from elementwise — neither is
            # CB-attached. Compiler inserts two independent intermediate DFBs.
            lhs = ttl.exp(av)
            rhs = ttl.neg(bv)
            o.store(lhs @ rhs)

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


def test_matmul_both_intermediates(device):
    """Both matmul operands from elementwise; two independent intermediate DFBs."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    out = to_l1(out_torch, device)

    expected = a_torch.float().exp() @ (-b_torch.float())

    matmul_both_intermediates_kernel(a, b, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result, expected, rtol=0.01, atol=0.5)


# --- two reduces on same intermediate, results feed elementwise sub ---
# The add result is materialized as an operand DFB for each reduce.
# Each reduce result feeds sub without a store, requiring result DFB
# materialization (not yet implemented, see #508).


@ttl.operation(grid=(1, 1))
def two_reduces_kernel(a, b, scaler, out):
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
            out_dfb.reserve() as o,
        ):
            added = ttl.add(av, bv)
            # Both reduces consume the same non-CB-attached add result.
            # Each reduce result feeds sub without a store, requiring
            # result DFB materialization (#508).
            sm = ttl.math.reduce_sum(added, sv, dims=[0, 1])
            mx = ttl.math.reduce_max(added, sv, dims=[0, 1])
            o.store(ttl.sub(sm, mx))

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


# --- multi-tile elementwise -> reduce (reproducer for #474) ---


TILE = 32
HEAD_TILES = 4


@ttl.operation(grid=(1, 1))
def multitile_mul_reduce_kernel(inp_a, inp_b, scaler, out):
    a_dfb = ttl.make_dataflow_buffer_like(inp_a, shape=(1, HEAD_TILES), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(inp_b, shape=(1, HEAD_TILES), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc:
            with a_dfb.wait() as av, b_dfb.wait() as bv:
                with out_dfb.reserve() as o:
                    o.store(ttl.math.reduce_sum(av * bv, sc, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with sc_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()
        with a_dfb.reserve() as blk:
            ttl.copy(inp_a[0:1, 0:HEAD_TILES], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(inp_b[0:1, 0:HEAD_TILES], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def test_multitile_mul_reduce(device):
    """Multi-tile elementwise mul feeds reduce_sum (regression for #474)."""
    from ttlang_test_utils import to_dram

    hd = HEAD_TILES * TILE
    a_torch = torch.randn(TILE, hd, dtype=torch.bfloat16)
    b_torch = torch.randn(TILE, hd, dtype=torch.bfloat16)
    scaler_torch = torch.ones(TILE, TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

    inp_a = to_dram(a_torch, device)
    inp_b = to_dram(b_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_dram(out_torch, device)

    expected = (a_torch.float() * b_torch.float()).sum()

    multitile_mul_reduce_kernel(inp_a, inp_b, scaler, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result[0, 0], expected, rtol=0.01, atol=0.5)


# --- intermediate DFB inside a loop with implicit pop ---
# Iterates over DRAM tiles in a loop. Each iteration: read tile, add with
# a running scaler, reduce_sum the add result. The compiler-allocated DFB
# for the add->reduce intermediate is created inside the loop body and
# must get correct push/pop from the sync pass within the with-scope.

STREAM_TILES = 4


@ttl.operation(grid=(1, 1))
def loop_reduce_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(STREAM_TILES):
            with (
                inp_dfb.wait() as x,
                scaler_dfb.wait() as s,
                out_dfb.reserve() as o,
            ):
                # add result is not CB-attached; compiler inserts
                # intermediate DFB inside the loop body.
                added = ttl.add(x, x)
                o.store(ttl.math.reduce_sum(added, s, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        for tile_idx in range(STREAM_TILES):
            with inp_dfb.reserve() as blk:
                ttl.copy(inp[0, tile_idx], blk).wait()
            with scaler_dfb.reserve() as blk:
                ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        for tile_idx in range(STREAM_TILES):
            with out_dfb.wait() as blk:
                ttl.copy(blk, out[0, tile_idx]).wait()


def test_loop_reduce(device):
    """Intermediate DFB inside a loop with implicit pop via with-statement."""
    from ttlang_test_utils import to_dram

    inp_torch = torch.randn(32, STREAM_TILES * 32, dtype=torch.bfloat16)
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, STREAM_TILES * 32, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_dram(out_torch, device)

    # Each iteration: reduce_sum(inp_tile + inp_tile) = 2 * sum(inp_tile).
    expected = torch.zeros(32, STREAM_TILES * 32)
    for tile_idx in range(STREAM_TILES):
        c0 = tile_idx * 32
        c1 = c0 + 32
        tile = inp_torch[:, c0:c1].float()
        expected[0, c0] = (tile + tile).sum()

    loop_reduce_kernel(inp, scaler, out)
    result = ttnn.to_torch(out).float()

    for tile_idx in range(STREAM_TILES):
        c0 = tile_idx * 32
        assert_allclose(result[0, c0], expected[0, c0], rtol=0.01, atol=0.5)


# --- intermediate DFB inside nested loops with conditional ---
# Streams tiles from a 2D grid. On even columns, the compute does
# add -> reduce_sum; on odd columns, mul -> reduce_sum. Both branches
# require a compiler-allocated intermediate DFB. The Python if/for
# unrolls into straight-line IR with different elementwise ops per
# iteration.

GRID_ROWS = 2
GRID_COLS = 2


@ttl.operation(grid=(1, 1))
def nested_loop_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(GRID_ROWS):
            for col in range(GRID_COLS):
                with (
                    inp_dfb.wait() as x,
                    scaler_dfb.wait() as s,
                    out_dfb.reserve() as o,
                ):
                    if col % 2 == 0:
                        intermediate = ttl.add(x, x)
                    else:
                        intermediate = ttl.mul(x, x)
                    o.store(ttl.math.reduce_sum(intermediate, s, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                with inp_dfb.reserve() as blk:
                    ttl.copy(inp[row, col], blk).wait()
                with scaler_dfb.reserve() as blk:
                    ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                with out_dfb.wait() as blk:
                    ttl.copy(blk, out[row, col]).wait()


@pytest.mark.skip(reason="DSL tracing error with if inside for loop (#510)")
def test_nested_loop_conditional(device):
    """Nested for-for-if with intermediate DFBs in both branches."""
    from ttlang_test_utils import to_dram

    inp_torch = torch.randn(GRID_ROWS * 32, GRID_COLS * 32, dtype=torch.bfloat16)
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(GRID_ROWS * 32, GRID_COLS * 32, dtype=torch.bfloat16)

    inp = to_dram(inp_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_dram(out_torch, device)

    expected = torch.zeros(GRID_ROWS * 32, GRID_COLS * 32)
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            r0, r1 = row * 32, (row + 1) * 32
            c0, c1 = col * 32, (col + 1) * 32
            tile = inp_torch[r0:r1, c0:c1].float()
            if col % 2 == 0:
                expected[r0, c0] = (tile + tile).sum()
            else:
                expected[r0, c0] = (tile * tile).sum()

    nested_loop_kernel(inp, scaler, out)
    result = ttnn.to_torch(out).float()

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            r0 = row * 32
            c0 = col * 32
            assert_allclose(result[r0, c0], expected[r0, c0], rtol=0.01, atol=1.0)


@pytest.mark.xfail(reason="Requires result DFB materialization (#508)")
def test_two_reduces(device):
    """Two reduces on same input, results feed sub; requires result DFBs (#508)."""
    a_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    b_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    scaler_torch = torch.ones(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    scaler = to_l1(scaler_torch, device)
    out = to_l1(out_torch, device)

    added = a_torch.float() + b_torch.float()
    expected = added.sum() - added.max()

    two_reduces_kernel(a, b, scaler, out)
    result = ttnn.to_torch(out).float()

    assert_allclose(result[0, 0], expected, rtol=0.01, atol=0.5)
