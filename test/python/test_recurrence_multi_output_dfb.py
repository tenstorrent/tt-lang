# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for loop-carried state plus compiler DFB outputs."""

import os
import sys

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32
N_ITERS = 4
K_TILES = 2
M_TILES = 2
N_BRANCH_NODES = 2
N_MULTI_OUTPUT_USES = 3


@ttl.operation(grid=(1, 1))
def reuse_intermediates(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, K_TILES), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(M_TILES, K_TILES), block_count=2)
    qk_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, M_TILES), block_count=2)
    max_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N_ITERS)

    @ttl.compute()
    def compute():
        for _ in range(N_ITERS):
            a_block = a_dfb.wait()
            b_block = b_dfb.wait()
            with qk_dfb.reserve() as qk_out:
                qk_out.store(a_block @ ttl.transpose(b_block))
            qk = qk_dfb.wait()
            with max_dfb.reserve() as max_out:
                max_out.store(ttl.math.reduce_max(qk, dims=[1]))
            max_block = max_dfb.wait()
            max_bcast = ttl.block.broadcast(max_block, dims=[1], shape=(1, M_TILES))
            exp_block = ttl.exp(ttl.sub(qk, max_bcast))
            with sum_dfb.reserve() as sum_out:
                sum_out.store(ttl.math.reduce_sum(exp_block, dims=[1]))
            sum_block = sum_dfb.wait()
            with out_dfb.reserve() as out_block:
                out_block.store(sum_block)

    @ttl.datamovement()
    def dm_read():
        for _ in range(N_ITERS):
            with a_dfb.reserve() as a_block:
                ttl.copy(a[0:1, 0:K_TILES], a_block).wait()
            with b_dfb.reserve() as b_block:
                ttl.copy(b[0:M_TILES, 0:K_TILES], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        for _ in range(N_ITERS):
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def recurrence_multi_output_intermediates(a, b, out):
    # This is the #666 reproducer: a running maximum recurrence with a derived
    # value that is stored back to state and consumed through a compiler DFB.
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, K_TILES), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(M_TILES, K_TILES), block_count=2)
    qk_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, M_TILES), block_count=2)
    max_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    state_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=N_ITERS)

    @ttl.compute()
    def compute():
        with state_dfb.reserve() as initial_state:
            initial_state.store(ttl.block.fill(-1e30, shape=(1, 1)))
        for _ in range(N_ITERS):
            a_block = a_dfb.wait()
            b_block = b_dfb.wait()
            with qk_dfb.reserve() as qk_out:
                qk_out.store(a_block @ ttl.transpose(b_block))
            qk = qk_dfb.wait()
            with max_dfb.reserve() as max_out:
                max_out.store(ttl.math.reduce_max(qk, dims=[1]))
            max_block = max_dfb.wait()
            state_old = state_dfb.wait()
            state_new = ttl.math.max(state_old, max_block)
            with state_dfb.reserve() as state_next:
                state_next.store(state_new)
            max_bcast = ttl.block.broadcast(state_new, dims=[1], shape=(1, M_TILES))
            exp_block = ttl.exp(ttl.sub(qk, max_bcast))
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
                ttl.copy(a[0:1, 0:K_TILES], a_block).wait()
            with b_dfb.reserve() as b_block:
                ttl.copy(b[0:M_TILES, 0:K_TILES], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        for _ in range(N_ITERS):
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def same_dfb_store_order(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=1)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=1)
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
            ttl.copy(a[0:1, 0:1], a_block).wait()
        with b_dfb.reserve() as b_block:
            ttl.copy(b[0:1, 0:1], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as first:
            ttl.copy(first, out[0:1, 0:1]).wait()
        with out_dfb.wait() as second:
            ttl.copy(second, out[0:1, 1:2]).wait()


@ttl.operation(grid=(1, 1))
def multi_output_out_of_order_consumers(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=1)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=1)
    sum_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=1)
    product_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=1)
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
            ttl.copy(a[0:1, 0:1], a_block).wait()
        with b_dfb.reserve() as b_block:
            ttl.copy(b[0:1, 0:1], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        for i in range(N_MULTI_OUTPUT_USES):
            with out_dfb.wait() as out_block:
                ttl.copy(out_block, out[0:1, i : i + 1]).wait()


@ttl.operation(grid=(N_BRANCH_NODES, 1))
def branch_intermediate_consumers(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
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
            ttl.copy(a[0, node_x], a_block).wait()
        with b_dfb.reserve() as b_block:
            ttl.copy(b[0, node_x], b_block).wait()

    @ttl.datamovement()
    def dm_write():
        node_x, _ = ttl.node(dims=2)
        with out_dfb.wait() as out_block:
            ttl.copy(out_block, out[0, node_x]).wait()


def _expected_sum(a_torch, b_torch):
    a_block = a_torch[0:TILE, 0 : K_TILES * TILE].float()
    b_block = b_torch[0 : M_TILES * TILE, 0 : K_TILES * TILE].float()
    qk = a_block @ b_block.T
    row_max = torch.max(qk, dim=1, keepdim=True).values
    return torch.sum(torch.exp(qk - row_max), dim=1, keepdim=True)


@pytest.mark.parametrize(
    "kernel",
    [reuse_intermediates, recurrence_multi_output_intermediates],
    ids=["reuse", "loop_state"],
)
@pytest.mark.requires_device
def test_recurrence_multi_output_dfb(kernel, device):
    torch.manual_seed(0)
    a_torch = torch.randn(TILE, K_TILES * TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(M_TILES * TILE, K_TILES * TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(out_torch, device)

    kernel(a, b, out)

    result = ttnn.to_torch(out).float()
    assert_pcc(_expected_sum(a_torch, b_torch), result[:, :1], threshold=0.99)


@pytest.mark.requires_device
def test_same_dfb_store_order(device):
    torch.manual_seed(1)
    a_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, 2 * TILE, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(out_torch, device)

    same_dfb_store_order(a, b, out)

    result = ttnn.to_torch(out).float()
    expected = torch.zeros_like(result)
    expected[:, :TILE] = a_torch.float() + b_torch.float()
    expected[:, TILE:] = a_torch.float() * b_torch.float()
    assert_pcc(expected, result, threshold=0.99)


@pytest.mark.requires_device
def test_multi_output_out_of_order_consumers(device):
    torch.manual_seed(3)
    a_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, N_MULTI_OUTPUT_USES * TILE, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(out_torch, device)

    multi_output_out_of_order_consumers(a, b, out)

    result = ttnn.to_torch(out).float()
    sum_block = a_torch.float() + b_torch.float()
    product_block = a_torch.float() * b_torch.float()
    expected = torch.zeros_like(result)
    expected[:, :1] = torch.sum(sum_block, dim=1, keepdim=True)
    expected[:, TILE : TILE + 1] = torch.sum(product_block, dim=1, keepdim=True)
    expected[:, 2 * TILE : 2 * TILE + 1] = torch.max(
        sum_block, dim=1, keepdim=True
    ).values

    assert_pcc(expected[:, :1], result[:, :1], threshold=0.99)
    assert_pcc(
        expected[:, TILE : TILE + 1],
        result[:, TILE : TILE + 1],
        threshold=0.99,
    )
    assert_pcc(
        expected[:, 2 * TILE : 2 * TILE + 1],
        result[:, 2 * TILE : 2 * TILE + 1],
        threshold=0.99,
    )


@pytest.mark.requires_device
def test_branch_intermediate_consumers(device):
    torch.manual_seed(2)
    a_torch = torch.randn(TILE, N_BRANCH_NODES * TILE, dtype=torch.bfloat16)
    b_torch = torch.randn(TILE, N_BRANCH_NODES * TILE, dtype=torch.bfloat16)
    out_torch = torch.zeros(TILE, N_BRANCH_NODES * TILE, dtype=torch.bfloat16)

    a = to_dram(a_torch, device)
    b = to_dram(b_torch, device)
    out = to_dram(out_torch, device)

    branch_intermediate_consumers(a, b, out)

    result = ttnn.to_torch(out).float()
    sum_block = a_torch.float() + b_torch.float()
    expected = torch.zeros_like(result)
    expected[:, :1] = torch.sum(sum_block[:, :TILE], dim=1, keepdim=True)
    expected[:, TILE : TILE + 1] = torch.max(
        sum_block[:, TILE:], dim=1, keepdim=True
    ).values
    assert_pcc(expected[:, :1], result[:, :1], threshold=0.99)
    assert_pcc(
        expected[:, TILE : TILE + 1],
        result[:, TILE : TILE + 1],
        threshold=0.99,
    )
