# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware coverage for matmul with leading batch dimensions."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_l1


@ttl.operation(grid=(1, 1))
def rank3_batched_matmul(a, b, out):
    """Multiply two batches of one-tile matrices."""
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(2, 1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_block,
            b_dfb.wait() as b_block,
            out_dfb.reserve() as out_block,
        ):
            out_block.store(a_block @ b_block)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as block:
            ttl.copy(a[0:2, 0:1, 0:1], block).wait()
        with b_dfb.reserve() as block:
            ttl.copy(b[0:2, 0:1, 0:1], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:2, 0:1, 0:1]).wait()


@ttl.operation(grid=(1, 1))
def rank4_batched_matmul_fused(a, b, bias, out):
    """Exercise two batch dimensions and a fused elementwise post-op."""
    block_shape = (2, 2, 1, 1)
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=block_shape, block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=block_shape, block_count=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=block_shape, block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=block_shape, block_count=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_block,
            b_dfb.wait() as b_block,
            bias_dfb.wait() as bias_block,
            out_dfb.reserve() as out_block,
        ):
            out_block.store((a_block @ b_block) + bias_block)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as block:
            ttl.copy(a[0:2, 0:2, 0:1, 0:1], block).wait()
        with b_dfb.reserve() as block:
            ttl.copy(b[0:2, 0:2, 0:1, 0:1], block).wait()
        with bias_dfb.reserve() as block:
            ttl.copy(bias[0:2, 0:2, 0:1, 0:1], block).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as block:
            ttl.copy(block, out[0:2, 0:2, 0:1, 0:1]).wait()


def _diagonal_batches(batch_shape, factors):
    identity = torch.eye(32, dtype=torch.bfloat16)
    result = torch.empty((*batch_shape, 32, 32), dtype=torch.bfloat16)
    for coordinates, factor in zip(
        torch.cartesian_prod(*(torch.arange(size) for size in batch_shape)),
        factors,
    ):
        index = tuple(int(coordinate) for coordinate in coordinates)
        result[index] = identity * factor
    return result


def test_rank3_batched_matmul(device):
    a_torch = torch.stack(
        [torch.eye(32, dtype=torch.bfloat16) * factor for factor in (1, 2)]
    )
    b_torch = torch.stack(
        [torch.eye(32, dtype=torch.bfloat16) * factor for factor in (3, 4)]
    )
    expected = torch.matmul(a_torch, b_torch)
    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    out = to_l1(torch.zeros_like(expected), device)

    rank3_batched_matmul(a, b, out)

    assert_allclose(ttnn.to_torch(out), expected)


def test_rank4_batched_matmul_fused(device):
    a_torch = _diagonal_batches((2, 2), (1, 2, 3, 4))
    b_torch = _diagonal_batches((2, 2), (5, 6, 7, 8))
    bias_torch = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2, 1, 1)
    bias_torch = bias_torch.expand(2, 2, 32, 32).clone()
    expected = torch.matmul(a_torch, b_torch) + bias_torch
    a = to_l1(a_torch, device)
    b = to_l1(b_torch, device)
    bias = to_l1(bias_torch, device)
    out = to_l1(torch.zeros_like(expected), device)

    rank4_batched_matmul_fused(a, b, bias, out)

    assert_allclose(ttnn.to_torch(out), expected)
