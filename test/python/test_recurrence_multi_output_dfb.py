# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for loop-carried state plus compiler DFB outputs."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32
N_ITERS = 4
K_TILES = 2
M_TILES = 2
N_BRANCH_NODES = 2
N_MULTI_OUTPUT_USES = 3
DTYPES = [torch.bfloat16, torch.float32]
COMPUTE_TILE_SIZES = [(16, 16), (16, 32), (32, 16), (32, 32)]


@ttl.operation(grid=(1, 1))
def reuse_intermediates(lhs, rhs, out):
    a_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, K_TILES), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(M_TILES, K_TILES), block_count=2)
    qk_reduce_dfb = ttl.make_dataflow_buffer_like(
        lhs, shape=(1, M_TILES), block_count=2
    )
    qk_exp_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, M_TILES), block_count=2)
    max_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N_ITERS)

    @ttl.compute()
    def compute():
        for _ in range(N_ITERS):
            a_block = a_dfb.wait()
            b_block = b_dfb.wait()
            qk_result = a_block @ ttl.transpose(b_block)
            # f32 reductions and elementwise consumers require different
            # unpack modes, so each strategy receives a separate DFB.
            with (
                qk_reduce_dfb.reserve() as qk_reduce_out,
                qk_exp_dfb.reserve() as qk_exp_out,
            ):
                qk_reduce_out.store(qk_result)
                qk_exp_out.store(qk_result)
            qk_reduce = qk_reduce_dfb.wait()
            with max_dfb.reserve() as max_out:
                max_out.store(ttl.math.reduce_max(qk_reduce, dims=[1]))
            max_block = max_dfb.wait()
            max_bcast = ttl.block.broadcast(max_block, dims=[1], shape=(1, M_TILES))
            qk_exp = qk_exp_dfb.wait()
            exp_block = ttl.exp(ttl.sub(qk_exp, max_bcast))
            with sum_dfb.reserve() as sum_out:
                sum_out.store(ttl.math.reduce_sum(exp_block, dims=[1]))
            sum_block = sum_dfb.wait()
            with out_dfb.reserve() as out_block:
                out_block.store(sum_block)

    @ttl.datamovement()
    def dm_read():
        for _ in range(N_ITERS):
            with a_dfb.reserve() as a_block:
                ttl.copy(lhs[0:1, 0:K_TILES], a_block).wait()
            with b_dfb.reserve() as b_block:
                ttl.copy(rhs[0:M_TILES, 0:K_TILES], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        for _ in range(N_ITERS):
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def recurrence_multi_output_intermediates(lhs, rhs, out):
    # This is the #666 reproducer: a running maximum recurrence with a derived
    # value that is stored back to state and consumed through a compiler DFB.
    a_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, K_TILES), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(M_TILES, K_TILES), block_count=2)
    qk_reduce_dfb = ttl.make_dataflow_buffer_like(
        lhs, shape=(1, M_TILES), block_count=2
    )
    qk_exp_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, M_TILES), block_count=2)
    max_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    state_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N_ITERS)

    @ttl.compute()
    def compute():
        with state_dfb.reserve() as initial_state:
            initial_state.store(
                ttl.block.fill(-1e30, shape=(1, 1), dtype=initial_state.dtype)
            )
        for _ in range(N_ITERS):
            a_block = a_dfb.wait()
            b_block = b_dfb.wait()
            qk_result = a_block @ ttl.transpose(b_block)
            # f32 reductions and elementwise consumers require different
            # unpack modes, so each strategy receives a separate DFB.
            with (
                qk_reduce_dfb.reserve() as qk_reduce_out,
                qk_exp_dfb.reserve() as qk_exp_out,
            ):
                qk_reduce_out.store(qk_result)
                qk_exp_out.store(qk_result)
            qk_reduce = qk_reduce_dfb.wait()
            with max_dfb.reserve() as max_out:
                max_out.store(ttl.math.reduce_max(qk_reduce, dims=[1]))
            max_block = max_dfb.wait()
            state_old = state_dfb.wait()
            state_new = ttl.math.max(state_old, max_block)
            with state_dfb.reserve() as state_next:
                state_next.store(state_new)
            max_bcast = ttl.block.broadcast(state_new, dims=[1], shape=(1, M_TILES))
            qk_exp = qk_exp_dfb.wait()
            exp_block = ttl.exp(ttl.sub(qk_exp, max_bcast))
            with sum_dfb.reserve() as sum_out:
                sum_out.store(ttl.math.reduce_sum(exp_block, dims=[1]))
            sum_block = sum_dfb.wait()
            with out_dfb.reserve() as out_block:
                out_block.store(sum_block)
        _ = state_dfb.wait()

    @ttl.datamovement()
    def dm_read():
        for _ in range(N_ITERS):
            with a_dfb.reserve() as a_block:
                ttl.copy(lhs[0:1, 0:K_TILES], a_block).wait()
            with b_dfb.reserve() as b_block:
                ttl.copy(rhs[0:M_TILES, 0:K_TILES], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        for _ in range(N_ITERS):
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def same_dfb_store_order(lhs, rhs, out):
    a_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    b_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        a_block = a_dfb.wait()
        b_block = b_dfb.wait()
        first = out_dfb.reserve()
        second = out_dfb.reserve()
        first.store(a_block + b_block)
        second.store(a_block * b_block)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as a_block:
            ttl.copy(lhs[0:1, 0:1], a_block).wait()
        with b_dfb.reserve() as b_block:
            ttl.copy(rhs[0:1, 0:1], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as first:
            ttl.copy(first, out[0:1, 0:1]).wait()
        with out_dfb.wait() as second:
            ttl.copy(second, out[0:1, 1:2]).wait()


@ttl.operation(grid=(1, 1))
def shared_publication_relocation(input_tensor, out):
    input_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)
    shared_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
    other_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        input_block = input_dfb.wait()
        inner = ttl.exp(input_block)
        outer = ttl.exp(inner)
        with shared_dfb.reserve() as shared_output:
            shared_output.store(inner)
            shared_output.store(outer)
        with other_dfb.reserve() as other_output:
            other_output.store(outer)

    @ttl.datamovement()
    def dm_read():
        with input_dfb.reserve() as input_block:
            ttl.copy(input_tensor[0:1, 0:1], input_block).wait()

    @ttl.datamovement()
    def dm_write():
        with shared_dfb.wait() as shared_output:
            ttl.copy(shared_output, out[0:1, 0:1]).wait()
        with other_dfb.wait() as other_output:
            ttl.copy(other_output, out[0:1, 1:2]).wait()


@ttl.operation(grid=(1, 1))
def multi_output_out_of_order_consumers(lhs, rhs, out):
    a_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    b_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=1)
    sum_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    product_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), block_count=N_MULTI_OUTPUT_USES
    )

    @ttl.compute()
    def compute():
        a_block = a_dfb.wait()
        b_block = b_dfb.wait()
        sum_block = a_block + b_block
        product_block = a_block * b_block
        with sum_dfb.reserve() as sum_out, product_dfb.reserve() as product_out:
            sum_out.store(sum_block)
            product_out.store(product_block)
        with out_dfb.reserve() as first_out:
            first_out.store(ttl.math.reduce_sum(sum_block, dims=[1]))
        with out_dfb.reserve() as second_out:
            second_out.store(ttl.math.reduce_sum(product_block, dims=[1]))
        with out_dfb.reserve() as third_out:
            third_out.store(ttl.math.reduce_max(sum_block, dims=[1]))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as a_block:
            ttl.copy(lhs[0:1, 0:1], a_block).wait()
        with b_dfb.reserve() as b_block:
            ttl.copy(rhs[0:1, 0:1], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        for output_index in range(N_MULTI_OUTPUT_USES):
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0:1, output_index : output_index + 1]).wait()


@ttl.operation(grid=(1, 1))
def published_value_mixed_consumers(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=1)
    published_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
    scaled_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        lhs_block = lhs_dfb.wait()
        rhs_block = rhs_dfb.wait()
        summed = lhs_block + rhs_block
        with published_dfb.reserve() as published:
            published.store(summed)
        row_sum = ttl.math.reduce_sum(summed, dims=[1])
        row_sum_bcast = ttl.block.broadcast(row_sum, dims=[1], shape=(1, 1))
        with scaled_dfb.reserve() as scaled:
            scaled.store(summed * row_sum_bcast)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0:1, 0:1], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0:1, 0:1], rhs_block).wait()

    @ttl.datamovement()
    def dm_write():
        with published_dfb.wait() as published:
            ttl.copy(published, out[0:1, 0:1]).wait()
        with scaled_dfb.wait() as scaled:
            ttl.copy(scaled, out[0:1, 1:2]).wait()


@ttl.operation(grid=(1, 1))
def shared_dfb_fixed_and_strategy_consumers(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        lhs_block = lhs_dfb.wait()
        rhs_block = rhs_dfb.wait()
        exponential = ttl.exp(lhs_block)
        summed = lhs_block + rhs_block
        with out_dfb.reserve() as out_block:
            out_block.store(exponential + summed)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0:1, 0:1], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0:1, 0:1], rhs_block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(N_BRANCH_NODES, 1))
def branch_intermediate_consumers(lhs, rhs, out):
    a_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_x, _ = ttl.node(dims=2)
        a_block = a_dfb.wait()
        b_block = b_dfb.wait()
        sum_block = a_block + b_block
        with sum_dfb.reserve() as sum_out:
            sum_out.store(sum_block)
        if node_x == 0:
            with out_dfb.reserve() as out_block:
                out_block.store(ttl.math.reduce_sum(sum_block, dims=[1]))
        else:
            with out_dfb.reserve() as out_block:
                out_block.store(ttl.math.reduce_max(sum_block, dims=[1]))
        _ = sum_dfb.wait()

    @ttl.datamovement()
    def dm_read():
        node_x, _ = ttl.node(dims=2)
        with a_dfb.reserve() as a_block:
            ttl.copy(lhs[0, node_x], a_block).wait()
        with b_dfb.reserve() as b_block:
            ttl.copy(rhs[0, node_x], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        node_x, _ = ttl.node(dims=2)
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, node_x]).wait()


@ttl.operation(grid=(N_BRANCH_NODES, 1))
def branch_elementwise_consumers(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        node_x, _ = ttl.node(dims=2)
        lhs_block = lhs_dfb.wait()
        rhs_block = rhs_dfb.wait()
        summed = lhs_block + rhs_block
        if node_x == 0:
            with out_dfb.reserve() as out_block:
                out_block.store(ttl.exp(summed))
        else:
            with out_dfb.reserve() as out_block:
                out_block.store(summed * summed)

    @ttl.datamovement()
    def dm_read():
        node_x, _ = ttl.node(dims=2)
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0, node_x], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0, node_x], rhs_block).wait()

    @ttl.datamovement()
    def dm_write():
        node_x, _ = ttl.node(dims=2)
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, node_x]).wait()


@ttl.operation(grid=(1, 1))
def published_value_two_elementwise_consumers(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=1)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=1)
    published_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
    exponential_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
    squared_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        lhs_block = lhs_dfb.wait()
        rhs_block = rhs_dfb.wait()
        summed = lhs_block + rhs_block
        with published_dfb.reserve() as published:
            published.store(summed)
        with exponential_dfb.reserve() as exponential:
            exponential.store(ttl.exp(summed))
        with squared_dfb.reserve() as squared:
            squared.store(summed * summed)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_block:
            ttl.copy(lhs[0, 0], lhs_block).wait()
        with rhs_dfb.reserve() as rhs_block:
            ttl.copy(rhs[0, 0], rhs_block).wait()

    @ttl.datamovement()
    def dm_write():
        with published_dfb.wait() as published:
            ttl.copy(published, out[0, 0]).wait()
        with exponential_dfb.wait() as exponential:
            ttl.copy(exponential, out[0, 1]).wait()
        with squared_dfb.wait() as squared:
            ttl.copy(squared, out[0, 2]).wait()


@ttl.operation(grid=(1, 1))
def reduce_elementwise_consumer(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        inp_block = inp_dfb.wait()
        reduced = ttl.math.reduce_sum(inp_block, dims=[1])
        with out_dfb.reserve() as out_block:
            out_block.store(ttl.exp(reduced))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as inp_block:
            ttl.copy(inp[0, 0], inp_block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, 0]).wait()


@ttl.operation(grid=(1, 1))
def reduce_two_publications_elementwise_consumer(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    reduced_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    exponential_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        inp_block = inp_dfb.wait()
        reduced = ttl.math.reduce_sum(inp_block, dims=[1])
        with reduced_dfb.reserve() as first_reduced:
            first_reduced.store(reduced)
        with reduced_dfb.reserve() as second_reduced:
            second_reduced.store(reduced)
        with exponential_dfb.reserve() as exponential:
            exponential.store(ttl.exp(reduced))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as inp_block:
            ttl.copy(inp[0, 0], inp_block).wait()

    @ttl.datamovement()
    def dm_write():
        with reduced_dfb.wait() as first_reduced:
            ttl.copy(first_reduced, out[0, 0]).wait()
        with reduced_dfb.wait() as second_reduced:
            ttl.copy(second_reduced, out[0, 1]).wait()
        with exponential_dfb.wait() as exponential:
            ttl.copy(exponential, out[0, 2]).wait()


@ttl.operation(grid=(1, 1))
def published_reduce_broadcast(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    reduced_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
    broadcast_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        inp_block = inp_dfb.wait()
        reduced = ttl.math.reduce_sum(inp_block, dims=[1])
        with reduced_dfb.reserve() as reduced_out:
            reduced_out.store(reduced)
        broadcast = ttl.block.broadcast(reduced, dims=[1], shape=(1, 1))
        with broadcast_dfb.reserve() as broadcast_out:
            broadcast_out.store(broadcast)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as inp_block:
            ttl.copy(inp[0, 0], inp_block).wait()

    @ttl.datamovement()
    def dm_write():
        with reduced_dfb.wait() as reduced:
            ttl.copy(reduced, out[0, 0]).wait()
        with broadcast_dfb.wait() as broadcast:
            ttl.copy(broadcast, out[0, 1]).wait()


def _expected_sum(lhs_torch, rhs_torch):
    lhs_block = lhs_torch[0:TILE, 0 : K_TILES * TILE].float()
    rhs_block = rhs_torch[0 : M_TILES * TILE, 0 : K_TILES * TILE].float()
    qk = lhs_block @ rhs_block.T
    row_max = torch.max(qk, dim=1, keepdim=True).values
    return torch.sum(torch.exp(qk - row_max), dim=1, keepdim=True)


def _pcc_threshold(dtype):
    return 0.99 if dtype == torch.bfloat16 else 0.999


@pytest.mark.parametrize(
    "kernel",
    [reuse_intermediates, recurrence_multi_output_intermediates],
    ids=["reuse", "loop_state"],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_recurrence_multi_output_dfb(kernel, dtype, device):
    torch.manual_seed(0)
    lhs_torch = torch.randn(TILE, K_TILES * TILE, dtype=dtype)
    rhs_torch = torch.randn(M_TILES * TILE, K_TILES * TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, TILE, dtype=dtype)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    kernel(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    assert_pcc(
        _expected_sum(lhs_torch, rhs_torch),
        result[:, :1],
        threshold=_pcc_threshold(dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_same_dfb_store_order(dtype, device):
    torch.manual_seed(1)
    lhs_torch = torch.randn(TILE, TILE, dtype=dtype)
    rhs_torch = torch.randn(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=dtype)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    same_dfb_store_order(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    expected = torch.zeros_like(result)
    expected[:, :TILE] = lhs_torch.float() + rhs_torch.float()
    expected[:, TILE:] = lhs_torch.float() * rhs_torch.float()
    assert_pcc(expected, result, threshold=_pcc_threshold(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_shared_publication_relocation(dtype, device):
    torch.manual_seed(4)
    input_torch = -torch.rand(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=dtype)

    input_tensor = to_dram(input_torch, device)
    out = to_dram(out_torch, device)

    shared_publication_relocation(input_tensor, out)

    result = ttnn.to_torch(out).float()
    outer = torch.exp(torch.exp(input_torch.float()))
    expected = torch.cat((outer, outer), dim=1)
    assert_pcc(expected, result, threshold=_pcc_threshold(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_multi_output_out_of_order_consumers(dtype, device):
    torch.manual_seed(3)
    lhs_torch = torch.randn(TILE, TILE, dtype=dtype)
    rhs_torch = torch.randn(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, N_MULTI_OUTPUT_USES * TILE, dtype=dtype)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    multi_output_out_of_order_consumers(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    sum_block = lhs_torch.float() + rhs_torch.float()
    product_block = lhs_torch.float() * rhs_torch.float()
    expected = torch.zeros_like(result)
    expected[:, :1] = torch.sum(sum_block, dim=1, keepdim=True)
    expected[:, TILE : TILE + 1] = torch.sum(product_block, dim=1, keepdim=True)
    expected[:, 2 * TILE : 2 * TILE + 1] = torch.max(
        sum_block, dim=1, keepdim=True
    ).values

    threshold = _pcc_threshold(dtype)
    assert_pcc(expected[:, :1], result[:, :1], threshold=threshold)
    assert_pcc(
        expected[:, TILE : TILE + 1],
        result[:, TILE : TILE + 1],
        threshold=threshold,
    )
    assert_pcc(
        expected[:, 2 * TILE : 2 * TILE + 1],
        result[:, 2 * TILE : 2 * TILE + 1],
        threshold=threshold,
    )


@pytest.mark.requires_device
def test_published_value_mixed_consumers(device):
    dtype = torch.bfloat16
    torch.manual_seed(5)
    lhs_torch = torch.randn(TILE, TILE, dtype=dtype)
    rhs_torch = torch.randn(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=dtype)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    published_value_mixed_consumers(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    summed = lhs_torch.float() + rhs_torch.float()
    expected = torch.cat((summed, summed * summed.sum(dim=1, keepdim=True)), dim=1)
    assert_pcc(expected, result, threshold=_pcc_threshold(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.parametrize(
    "tile_hw",
    COMPUTE_TILE_SIZES,
    ids=[f"{height}x{width}" for height, width in COMPUTE_TILE_SIZES],
)
@pytest.mark.requires_device
def test_shared_dfb_fixed_and_strategy_consumers(dtype, tile_hw, device):
    torch.manual_seed(6)
    tile_height, tile_width = tile_hw
    lhs_torch = torch.randn(tile_height, tile_width, dtype=dtype)
    rhs_torch = torch.randn(tile_height, tile_width, dtype=dtype)
    out_torch = torch.zeros(tile_height, tile_width, dtype=dtype)

    lhs = to_dram(lhs_torch, device, tile=tile_hw)
    rhs = to_dram(rhs_torch, device, tile=tile_hw)
    out = to_dram(out_torch, device, tile=tile_hw)

    shared_dfb_fixed_and_strategy_consumers(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    expected = torch.exp(lhs_torch.float()) + lhs_torch.float() + rhs_torch.float()
    assert_pcc(expected, result, threshold=_pcc_threshold(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_branch_intermediate_consumers(dtype, device):
    torch.manual_seed(2)
    lhs_torch = torch.randn(TILE, N_BRANCH_NODES * TILE, dtype=dtype)
    rhs_torch = torch.randn(TILE, N_BRANCH_NODES * TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, N_BRANCH_NODES * TILE, dtype=dtype)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    branch_intermediate_consumers(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    sum_block = lhs_torch.float() + rhs_torch.float()
    expected = torch.zeros_like(result)
    expected[:, :1] = torch.sum(sum_block[:, :TILE], dim=1, keepdim=True)
    expected[:, TILE : TILE + 1] = torch.max(
        sum_block[:, TILE:], dim=1, keepdim=True
    ).values
    threshold = _pcc_threshold(dtype)
    assert_pcc(expected[:, :1], result[:, :1], threshold=threshold)
    assert_pcc(
        expected[:, TILE : TILE + 1],
        result[:, TILE : TILE + 1],
        threshold=threshold,
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_branch_elementwise_consumers(dtype, device):
    torch.manual_seed(6)
    lhs_torch = 0.1 * torch.randn(TILE, N_BRANCH_NODES * TILE, dtype=dtype)
    rhs_torch = 0.1 * torch.randn(TILE, N_BRANCH_NODES * TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, N_BRANCH_NODES * TILE, dtype=dtype)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    branch_elementwise_consumers(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    summed = lhs_torch.float() + rhs_torch.float()
    expected = torch.cat(
        (torch.exp(summed[:, :TILE]), summed[:, TILE:] * summed[:, TILE:]),
        dim=1,
    )
    assert_pcc(expected, result, threshold=_pcc_threshold(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_published_value_two_elementwise_consumers(dtype, device):
    torch.manual_seed(7)
    lhs_torch = 0.1 * torch.randn(TILE, TILE, dtype=dtype)
    rhs_torch = 0.1 * torch.randn(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, 3 * TILE, dtype=dtype)

    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    published_value_two_elementwise_consumers(lhs, rhs, out)

    result = ttnn.to_torch(out).float()
    summed = lhs_torch.float() + rhs_torch.float()
    expected = torch.cat((summed, torch.exp(summed), summed * summed), dim=1)
    assert_pcc(expected, result, threshold=_pcc_threshold(dtype))


@pytest.mark.parametrize(
    "kernel, output_tile_count",
    [
        (reduce_elementwise_consumer, 1),
        (reduce_two_publications_elementwise_consumer, 3),
    ],
    ids=["unstored", "two_publications"],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_reduce_elementwise_consumer(kernel, output_tile_count, dtype, device):
    torch.manual_seed(8)
    inp_torch = -0.01 * torch.rand(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, output_tile_count * TILE, dtype=dtype)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)

    result = ttnn.to_torch(out).float()
    reduced = torch.sum(inp_torch.float(), dim=1, keepdim=True)
    expected_columns = [torch.exp(reduced)]
    if output_tile_count == 3:
        expected_columns = [reduced, reduced, torch.exp(reduced)]
    for tile_index, expected in enumerate(expected_columns):
        column = result[:, tile_index * TILE : tile_index * TILE + 1]
        assert_pcc(expected, column, threshold=_pcc_threshold(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "f32"])
@pytest.mark.requires_device
def test_published_reduce_broadcast(dtype, device):
    torch.manual_seed(9)
    inp_torch = torch.randn(TILE, TILE, dtype=dtype)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=dtype)

    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    published_reduce_broadcast(inp, out)

    result = ttnn.to_torch(out).float()
    reduced = torch.sum(inp_torch.float(), dim=1, keepdim=True)
    threshold = _pcc_threshold(dtype)
    assert_pcc(reduced, result[:, :1], threshold=threshold)
    assert_pcc(reduced.expand(-1, TILE), result[:, TILE:], threshold=threshold)
