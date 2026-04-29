# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression for #527: self-rebound tensor values must leave scf.for loops."""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32


def _make_loop_carried_add_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a, weights, recv, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        weights_dfb = ttl.make_dataflow_buffer_like(
            weights, shape=(1, 1), block_count=2
        )
        partial_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(recv, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, weights_dfb.wait() as weights_blk:
                with partial_dfb.reserve() as partial_blk:
                    partial_blk.store(a_blk @ weights_blk)

            with partial_dfb.wait() as acc:
                for _ in range(1):
                    with recv_dfb.wait() as recv_blk:
                        acc = acc + recv_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)

        @ttl.datamovement()
        def reader():
            with a_dfb.reserve() as a_blk:
                ttl.copy(a[0:1, 0:1], a_blk).wait()
            with weights_dfb.reserve() as weights_blk:
                ttl.copy(weights[0:1, 0:1], weights_blk).wait()
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(recv[0:1, 0:1], recv_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


def _make_loop_carried_relu_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(initial, bias, out):
        initial_dfb = ttl.make_dataflow_buffer_like(
            initial, shape=(1, 1), block_count=2
        )
        bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with initial_dfb.wait() as state:
                for _ in range(1):
                    with bias_dfb.wait() as bias_blk:
                        state = ttl.math.relu(state + bias_blk)

                with out_dfb.reserve() as out_blk:
                    out_blk.store(state)

        @ttl.datamovement()
        def reader():
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(initial[0:1, 0:1], initial_blk).wait()
            with bias_dfb.reserve() as bias_blk:
                ttl.copy(bias[0:1, 0:1], bias_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
def test_self_rebound_add_result_is_carried_out_of_loop(device):
    kernel = _make_loop_carried_add_kernel()

    a = torch.ones((TILE, TILE), dtype=torch.bfloat16)
    weights = torch.ones((TILE, TILE), dtype=torch.bfloat16)
    recv = torch.full((TILE, TILE), 2.0, dtype=torch.bfloat16)
    out = torch.zeros((TILE, TILE), dtype=torch.bfloat16)

    expected = a.float() @ weights.float() + recv.float()

    a_dev = to_dram(a, device)
    weights_dev = to_dram(weights, device)
    recv_dev = to_dram(recv, device)
    out_dev = to_dram(out, device)

    kernel(a_dev, weights_dev, recv_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.requires_device
def test_non_add_tensor_recurrence_is_carried_out_of_loop(device):
    kernel = _make_loop_carried_relu_kernel()

    initial = torch.full((TILE, TILE), -3.0, dtype=torch.bfloat16)
    bias = torch.full((TILE, TILE), 5.0, dtype=torch.bfloat16)
    out = torch.zeros((TILE, TILE), dtype=torch.bfloat16)

    expected = torch.relu(initial.float() + bias.float())

    initial_dev = to_dram(initial, device)
    bias_dev = to_dram(bias, device)
    out_dev = to_dram(out, device)

    kernel(initial_dev, bias_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)
