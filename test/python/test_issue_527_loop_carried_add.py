# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression for #527: self-rebound tensor values must leave scf.for loops."""


import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

TILE = 32


N_ITERS = 3


def _make_loop_carried_add_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a, weights, recv, out):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
        weights_dfb = ttl.make_dataflow_buffer_like(
            weights, shape=(1, 1), block_count=2
        )
        partial_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(
            recv, shape=(1, 1), block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, weights_dfb.wait() as weights_blk:
                with partial_dfb.reserve() as partial_blk:
                    partial_blk.store(a_blk @ weights_blk)

            with partial_dfb.wait() as acc:
                for _ in range(N_ITERS):
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
            for _ in range(N_ITERS):
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
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias, shape=(1, 1), block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with initial_dfb.wait() as state:
                for _ in range(N_ITERS):
                    with bias_dfb.wait() as bias_blk:
                        state = ttl.math.relu(state + bias_blk)

                with out_dfb.reserve() as out_blk:
                    out_blk.store(state)

        @ttl.datamovement()
        def reader():
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(initial[0:1, 0:1], initial_blk).wait()
            for _ in range(N_ITERS):
                with bias_dfb.reserve() as bias_blk:
                    ttl.copy(bias[0:1, 0:1], bias_blk).wait()

        @ttl.datamovement()
        def writer():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()

    return kernel


_DTYPE_TOL = {
    torch.bfloat16: dict(rtol=5e-2, atol=1.0),
    torch.float32: dict(rtol=1e-3, atol=1e-3),
}


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_self_rebound_add_result_is_carried_out_of_loop(device, dtype):
    kernel = _make_loop_carried_add_kernel()

    a = torch.ones((TILE, TILE), dtype=dtype)
    weights = torch.ones((TILE, TILE), dtype=dtype)
    recv = torch.full((TILE, TILE), 2.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)

    expected = a.float() @ weights.float() + N_ITERS * recv.float()

    a_dev = to_dram(a, device)
    weights_dev = to_dram(weights, device)
    recv_dev = to_dram(recv, device)
    out_dev = to_dram(out, device)

    kernel(a_dev, weights_dev, recv_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result.float(), expected.float(), **_DTYPE_TOL[dtype])


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_non_add_tensor_recurrence_is_carried_out_of_loop(device, dtype):
    kernel = _make_loop_carried_relu_kernel()

    initial = torch.full((TILE, TILE), -3.0, dtype=dtype)
    bias = torch.full((TILE, TILE), 5.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)

    state = initial.float()
    for _ in range(N_ITERS):
        state = torch.relu(state + bias.float())
    expected = state

    initial_dev = to_dram(initial, device)
    bias_dev = to_dram(bias, device)
    out_dev = to_dram(out, device)

    kernel(initial_dev, bias_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result.float(), expected.float(), **_DTYPE_TOL[dtype])


def _make_tuple_target_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, b_seed, delta, out_a, out_b):
        a_cb = ttl.make_dataflow_buffer_like(a_seed, shape=(1, 1), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(b_seed, shape=(1, 1), block_count=2)
        delta_cb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS
        )
        out_a_cb = ttl.make_dataflow_buffer_like(out_a, shape=(1, 1), block_count=2)
        out_b_cb = ttl.make_dataflow_buffer_like(out_b, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with a_cb.wait() as a, b_cb.wait() as b:
                for _ in range(N_ITERS):
                    with delta_cb.wait() as d:
                        a, b = a + d, b + d
                with out_a_cb.reserve() as oa:
                    oa.store(a)
                with out_b_cb.reserve() as ob:
                    ob.store(b)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(a_seed[0:1, 0:1], blk).wait()
            with b_cb.reserve() as blk:
                ttl.copy(b_seed[0:1, 0:1], blk).wait()
            for _ in range(N_ITERS):
                with delta_cb.reserve() as blk:
                    ttl.copy(delta[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_a_cb.wait() as blk:
                ttl.copy(blk, out_a[0:1, 0:1]).wait()
            with out_b_cb.wait() as blk:
                ttl.copy(blk, out_b[0:1, 0:1]).wait()

    return kernel


N_COND_ITERS = 4
COND_THRESHOLD = 2


def _make_conditional_rebind_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(initial, bias, out):
        initial_cb = ttl.make_dataflow_buffer_like(initial, shape=(1, 1), block_count=2)
        bias_cb = ttl.make_dataflow_buffer_like(
            bias, shape=(1, 1), block_count=N_COND_ITERS
        )
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with initial_cb.wait() as x:
                for i in range(N_COND_ITERS):
                    with bias_cb.wait() as bias_blk:
                        if i < COND_THRESHOLD:
                            x = x + bias_blk
                with out_cb.reserve() as o:
                    o.store(x)

        @ttl.datamovement()
        def reader():
            with initial_cb.reserve() as blk:
                ttl.copy(initial[0:1, 0:1], blk).wait()
            for _ in range(N_COND_ITERS):
                with bias_cb.reserve() as blk:
                    ttl.copy(bias[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.xfail(
    strict=True,
    reason="ttl-assign-dst does not descend into nested regions (#587), so "
    "tile ops inside scf.if fail legalization in convert-ttl-to-ttkernel. "
    "AST emission and materialize-loop-state are correct (see lit case "
    "conditional_recurrence in materialize_loop_state.mlir).",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_conditional_rebind_inside_loop_carries_through_scf_if(device, dtype):
    kernel = _make_conditional_rebind_kernel()

    initial = torch.full((TILE, TILE), 0.0, dtype=dtype)
    bias = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)

    expected = initial.float() + COND_THRESHOLD * bias.float()

    initial_dev = to_dram(initial, device)
    bias_dev = to_dram(bias, device)
    out_dev = to_dram(out, device)

    kernel(initial_dev, bias_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result.float(), expected.float(), **_DTYPE_TOL[dtype])


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_tuple_target_self_rebind_carries_both(device, dtype):
    kernel = _make_tuple_target_kernel()

    a_seed = torch.full((TILE, TILE), 1.0, dtype=dtype)
    b_seed = torch.full((TILE, TILE), 10.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out_a = torch.zeros((TILE, TILE), dtype=dtype)
    out_b = torch.zeros((TILE, TILE), dtype=dtype)

    expected_a = a_seed.float() + N_ITERS * delta.float()
    expected_b = b_seed.float() + N_ITERS * delta.float()

    a_dev = to_dram(a_seed, device)
    b_dev = to_dram(b_seed, device)
    delta_dev = to_dram(delta, device)
    out_a_dev = to_dram(out_a, device)
    out_b_dev = to_dram(out_b, device)

    kernel(a_dev, b_dev, delta_dev, out_a_dev, out_b_dev)
    ttnn.synchronize_device(device)

    result_a = ttnn.to_torch(out_a_dev).float()
    result_b = ttnn.to_torch(out_b_dev).float()
    assert_allclose(result_a, expected_a.float(), **_DTYPE_TOL[dtype])
    assert_allclose(result_b, expected_b.float(), **_DTYPE_TOL[dtype])
