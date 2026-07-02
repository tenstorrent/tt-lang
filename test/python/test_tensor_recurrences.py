# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor recurrences carried through control flow.

Covers loop-carried recurrences (`acc = acc + x`), `+=` rewrite, tuple
targets, multi-accumulator and multi-tile recurrences, zero-trip loops,
and conditional rebinds. The core regression is #527 (`acc = acc +
recv.wait()` silently dropped its loop-carried value); the rest
exercise `ttl-materialize-loop-state`, augmented-assignment rewriting,
and existing DFB-attached block `+=` behavior.
"""

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


def _make_direct_loop_carried_add_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(initial, delta, out):
        initial_dfb = ttl.make_dataflow_buffer_like(
            initial, shape=(1, 1), block_count=2
        )
        delta_dfb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS
        )
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with initial_dfb.wait() as acc:
                for _ in range(N_ITERS):
                    with delta_dfb.wait() as delta_blk:
                        acc = acc + delta_blk

                with out_dfb.reserve() as out_blk:
                    out_blk.store(acc)

        @ttl.datamovement()
        def reader():
            with initial_dfb.reserve() as initial_blk:
                ttl.copy(initial[0:1, 0:1], initial_blk).wait()
            for _ in range(N_ITERS):
                with delta_dfb.reserve() as delta_blk:
                    ttl.copy(delta[0:1, 0:1], delta_blk).wait()

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


def _run_io_kernel(
    kernel,
    in_tensors,
    out_zeros,
    expected_list,
    dtype,
    device,
):
    """Common test runner: move inputs and outputs to device, invoke the
    kernel, then assert each output tensor matches its expected value."""
    in_devs = [to_dram(t, device) for t in in_tensors]
    out_devs = [to_dram(t, device) for t in out_zeros]
    kernel(*in_devs, *out_devs)
    ttnn.synchronize_device(device)
    for out_dev, expected in zip(out_devs, expected_list):
        result = ttnn.to_torch(out_dev).float()
        assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])


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
def test_direct_loop_carried_add(device, dtype):
    """Direct additive recurrences are materialized through DFB state."""
    initial = torch.full((TILE, TILE), 4.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 2.0, dtype=dtype)
    expected = initial.float() + N_ITERS * delta.float()
    _run_io_kernel(
        _make_direct_loop_carried_add_kernel(),
        in_tensors=[initial, delta],
        out_zeros=[torch.zeros((TILE, TILE), dtype=dtype)],
        expected_list=[expected],
        dtype=dtype,
        device=device,
    )


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


def _make_three_accumulator_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, b_seed, c_seed, delta, out_a, out_b, out_c):
        a_cb = ttl.make_dataflow_buffer_like(a_seed, shape=(1, 1), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(b_seed, shape=(1, 1), block_count=2)
        c_cb = ttl.make_dataflow_buffer_like(c_seed, shape=(1, 1), block_count=2)
        delta_cb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS
        )
        out_a_cb = ttl.make_dataflow_buffer_like(out_a, shape=(1, 1), block_count=2)
        out_b_cb = ttl.make_dataflow_buffer_like(out_b, shape=(1, 1), block_count=2)
        out_c_cb = ttl.make_dataflow_buffer_like(out_c, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with a_cb.wait() as a, b_cb.wait() as b, c_cb.wait() as c:
                for _ in range(N_ITERS):
                    with delta_cb.wait() as d:
                        a, b, c = a + d, b + d, c + d
                with out_a_cb.reserve() as oa:
                    oa.store(a)
                with out_b_cb.reserve() as ob:
                    ob.store(b)
                with out_c_cb.reserve() as oc:
                    oc.store(c)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(a_seed[0:1, 0:1], blk).wait()
            with b_cb.reserve() as blk:
                ttl.copy(b_seed[0:1, 0:1], blk).wait()
            with c_cb.reserve() as blk:
                ttl.copy(c_seed[0:1, 0:1], blk).wait()
            for _ in range(N_ITERS):
                with delta_cb.reserve() as blk:
                    ttl.copy(delta[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_a_cb.wait() as blk:
                ttl.copy(blk, out_a[0:1, 0:1]).wait()
            with out_b_cb.wait() as blk:
                ttl.copy(blk, out_b[0:1, 0:1]).wait()
            with out_c_cb.wait() as blk:
                ttl.copy(blk, out_c[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_three_accumulators_in_one_loop(device, dtype):
    """Covers three DFB-backed tensor recurrences in one loop."""
    kernel = _make_three_accumulator_kernel()

    a_seed = torch.full((TILE, TILE), 1.0, dtype=dtype)
    b_seed = torch.full((TILE, TILE), 10.0, dtype=dtype)
    c_seed = torch.full((TILE, TILE), 100.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out_a = torch.zeros((TILE, TILE), dtype=dtype)
    out_b = torch.zeros((TILE, TILE), dtype=dtype)
    out_c = torch.zeros((TILE, TILE), dtype=dtype)

    expected_a = a_seed.float() + N_ITERS * delta.float()
    expected_b = b_seed.float() + N_ITERS * delta.float()
    expected_c = c_seed.float() + N_ITERS * delta.float()

    a_dev = to_dram(a_seed, device)
    b_dev = to_dram(b_seed, device)
    c_dev = to_dram(c_seed, device)
    delta_dev = to_dram(delta, device)
    out_a_dev = to_dram(out_a, device)
    out_b_dev = to_dram(out_b, device)
    out_c_dev = to_dram(out_c, device)

    kernel(a_dev, b_dev, c_dev, delta_dev, out_a_dev, out_b_dev, out_c_dev)
    ttnn.synchronize_device(device)

    result_a = ttnn.to_torch(out_a_dev).float()
    result_b = ttnn.to_torch(out_b_dev).float()
    result_c = ttnn.to_torch(out_c_dev).float()
    assert_allclose(result_a, expected_a.float(), **_DTYPE_TOL[dtype])
    assert_allclose(result_b, expected_b.float(), **_DTYPE_TOL[dtype])
    assert_allclose(result_c, expected_c.float(), **_DTYPE_TOL[dtype])


MULTI_TILE_SHAPE = (2, 2)
MULTI_TILE_ROWS = MULTI_TILE_SHAPE[0] * TILE
MULTI_TILE_COLS = MULTI_TILE_SHAPE[1] * TILE


def _make_multi_tile_block_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, delta, out):
        a_cb = ttl.make_dataflow_buffer_like(
            a_seed, shape=MULTI_TILE_SHAPE, block_count=2
        )
        delta_cb = ttl.make_dataflow_buffer_like(
            delta, shape=MULTI_TILE_SHAPE, block_count=N_ITERS
        )
        out_cb = ttl.make_dataflow_buffer_like(
            out, shape=MULTI_TILE_SHAPE, block_count=2
        )

        @ttl.compute()
        def compute():
            with a_cb.wait() as a:
                for _ in range(N_ITERS):
                    with delta_cb.wait() as d:
                        a = a + d
                with out_cb.reserve() as o:
                    o.store(a)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(
                    a_seed[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]], blk
                ).wait()
            for _ in range(N_ITERS):
                with delta_cb.reserve() as blk:
                    ttl.copy(
                        delta[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]], blk
                    ).wait()

        @ttl.datamovement()
        def writer():
            with out_cb.wait() as blk:
                ttl.copy(
                    blk, out[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]]
                ).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_multi_tile_block_recurrence(device, dtype):
    """Verifies recurrence carries a multi-tile block (shape=(2, 2)), not just a single tile."""
    kernel = _make_multi_tile_block_kernel()

    a_seed = torch.full((MULTI_TILE_ROWS, MULTI_TILE_COLS), 1.0, dtype=dtype)
    delta = torch.full((MULTI_TILE_ROWS, MULTI_TILE_COLS), 2.0, dtype=dtype)
    out = torch.zeros((MULTI_TILE_ROWS, MULTI_TILE_COLS), dtype=dtype)

    expected = a_seed.float() + N_ITERS * delta.float()

    a_dev = to_dram(a_seed, device)
    delta_dev = to_dram(delta, device)
    out_dev = to_dram(out, device)

    kernel(a_dev, delta_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])


def _make_aug_assign_non_block_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, delta, out):
        a_cb = ttl.make_dataflow_buffer_like(a_seed, shape=(1, 1), block_count=2)
        # Need N_ITERS + 1 delta tiles: one for the pre-loop seed, then one
        # per loop iteration.
        delta_cb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS + 1
        )
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with a_cb.wait() as a, delta_cb.wait() as d_init:
                # `acc` is the result of ttl.add, not an attach -> not a
                # block. `acc += d` inside the loop must therefore rewrite
                # to `acc = acc + d` (loop-carried recurrence), not invoke
                # __iadd__.
                acc = a + d_init
                for _ in range(N_ITERS):
                    with delta_cb.wait() as d:
                        acc += d
                with out_cb.reserve() as o:
                    o.store(acc)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(a_seed[0:1, 0:1], blk).wait()
            for _ in range(N_ITERS + 1):
                with delta_cb.reserve() as blk:
                    ttl.copy(delta[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_aug_assign_on_non_block_rewrites_to_loop_carried(device, dtype):
    """`acc += d` on a plain tensor (not a reserve block) is rewritten by
    the AST visitor to `acc = acc + d` so it lowers through
    ttl-materialize-loop-state."""
    kernel = _make_aug_assign_non_block_kernel()

    a_seed = torch.full((TILE, TILE), 1.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 2.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)

    # acc starts at a + delta = 3, then N_ITERS more deltas added.
    expected = a_seed.float() + (N_ITERS + 1) * delta.float()

    a_dev = to_dram(a_seed, device)
    delta_dev = to_dram(delta, device)
    out_dev = to_dram(out, device)

    kernel(a_dev, delta_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])


def _make_multi_target_aug_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, b_seed, delta, out_a, out_b):
        a_cb = ttl.make_dataflow_buffer_like(a_seed, shape=(1, 1), block_count=2)
        b_cb = ttl.make_dataflow_buffer_like(b_seed, shape=(1, 1), block_count=2)
        delta_cb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS + 1
        )
        out_a_cb = ttl.make_dataflow_buffer_like(out_a, shape=(1, 1), block_count=2)
        out_b_cb = ttl.make_dataflow_buffer_like(out_b, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with a_cb.wait() as a, b_cb.wait() as b, delta_cb.wait() as d0:
                acc1 = a + d0
                acc2 = b + d0
                for _ in range(N_ITERS):
                    with delta_cb.wait() as d:
                        acc1 += d
                        acc2 += d
                with out_a_cb.reserve() as o:
                    o.store(acc1)
                with out_b_cb.reserve() as o:
                    o.store(acc2)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(a_seed[0:1, 0:1], blk).wait()
            with b_cb.reserve() as blk:
                ttl.copy(b_seed[0:1, 0:1], blk).wait()
            for _ in range(N_ITERS + 1):
                with delta_cb.reserve() as blk:
                    ttl.copy(delta[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_a_cb.wait() as blk:
                ttl.copy(blk, out_a[0:1, 0:1]).wait()
            with out_b_cb.wait() as blk:
                ttl.copy(blk, out_b[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_aug_assign_multi_target_in_loop(device, dtype):
    """Two plain-tensor accumulators each rebound with `+=` in the same
    loop: exercises the AugAssign rewrite alongside multi-iter_arg
    emission and multiple DFB-attached accumulating stores."""
    a_seed = torch.full((TILE, TILE), 1.0, dtype=dtype)
    b_seed = torch.full((TILE, TILE), 10.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 1.0, dtype=dtype)
    expected_a = a_seed.float() + (N_ITERS + 1) * delta.float()
    expected_b = b_seed.float() + (N_ITERS + 1) * delta.float()
    _run_io_kernel(
        _make_multi_target_aug_kernel(),
        in_tensors=[a_seed, b_seed, delta],
        out_zeros=[
            torch.zeros((TILE, TILE), dtype=dtype),
            torch.zeros((TILE, TILE), dtype=dtype),
        ],
        expected_list=[expected_a, expected_b],
        dtype=dtype,
        device=device,
    )


def _make_aug_outside_loop_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, delta, out):
        a_cb = ttl.make_dataflow_buffer_like(a_seed, shape=(1, 1), block_count=2)
        delta_fpu_cb = ttl.make_dataflow_buffer_like(delta, shape=(1, 1), block_count=2)
        delta_sfpu_cb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=2
        )
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with (
                a_cb.wait() as a,
                delta_fpu_cb.wait() as d_fpu,
                delta_sfpu_cb.wait() as d_sfpu,
            ):
                acc = a + d_fpu
                # `+=` outside any loop: the rewrite produces a plain
                # `acc = acc + d` with no recurrence machinery involved.
                acc += d_sfpu
                with out_cb.reserve() as o:
                    o.store(acc)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(a_seed[0:1, 0:1], blk).wait()
            with delta_fpu_cb.reserve() as blk:
                ttl.copy(delta[0:1, 0:1], blk).wait()
            with delta_sfpu_cb.reserve() as blk:
                ttl.copy(delta[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_aug_assign_outside_loop(device, dtype):
    """`acc += d` outside any loop must lower to a single ttl.add, not
    accidentally synthesize a scf.for."""
    a_seed = torch.full((TILE, TILE), 1.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 2.0, dtype=dtype)
    expected = a_seed.float() + 2 * delta.float()
    _run_io_kernel(
        _make_aug_outside_loop_kernel(),
        in_tensors=[a_seed, delta],
        out_zeros=[torch.zeros((TILE, TILE), dtype=dtype)],
        expected_list=[expected],
        dtype=dtype,
        device=device,
    )


def _make_sub_aug_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, delta, out):
        a_cb = ttl.make_dataflow_buffer_like(a_seed, shape=(1, 1), block_count=2)
        delta_cb = ttl.make_dataflow_buffer_like(
            delta, shape=(1, 1), block_count=N_ITERS + 1
        )
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with a_cb.wait() as a, delta_cb.wait() as d_init:
                acc = a - d_init
                for _ in range(N_ITERS):
                    with delta_cb.wait() as d:
                        acc -= d
                with out_cb.reserve() as o:
                    o.store(acc)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(a_seed[0:1, 0:1], blk).wait()
            for _ in range(N_ITERS + 1):
                with delta_cb.reserve() as blk:
                    ttl.copy(delta[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_aug_assign_sub_op(device, dtype):
    """`-=` on a plain tensor is rewritten the same way as `+=`; it lowers
    through ttl.sub instead of DFB-attached in-place addition.
    Final value: a - d_init - N_ITERS * d."""
    a_seed = torch.full((TILE, TILE), 100.0, dtype=dtype)
    delta = torch.full((TILE, TILE), 2.0, dtype=dtype)
    expected = a_seed.float() - delta.float() - N_ITERS * delta.float()
    _run_io_kernel(
        _make_sub_aug_kernel(),
        in_tensors=[a_seed, delta],
        out_zeros=[torch.zeros((TILE, TILE), dtype=dtype)],
        expected_list=[expected],
        dtype=dtype,
        device=device,
    )


def _make_block_aug_in_loop_kernel():
    """Smallest reproducer for the iter-arg-shadows-block regression:
    `out_blk += x` on a reserve block whose `with:` scope encloses an
    `scf.for` loop. The collector must NOT register `out_blk` as a
    loop-carried iter_arg; if it does, the block target is shadowed to
    an scf.for BlockArgument and `+=` falls into the plain-tensor
    rewrite (`ttl.add` producing an SSA value) instead of the
    block `__iadd__` accumulating store, leaving the DFB at its initial fill."""

    @ttl.operation(grid=(1, 1))
    def kernel(x, out):
        x_cb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=N_ITERS)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with out_cb.reserve() as out_blk:
                out_blk.store(
                    ttl.block.fill(0, shape=out_blk.shape, dtype=out_blk.dtype)
                )
                for _ in range(N_ITERS):
                    with x_cb.wait() as xj:
                        out_blk += xj

        @ttl.datamovement()
        def reader():
            for _ in range(N_ITERS):
                with x_cb.reserve() as blk:
                    ttl.copy(x[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_block_aug_assign_in_loop_uses_l1_acc(device, dtype):
    """Regression for #540 review: `out_blk += x` inside a for loop on a
    reserve block carried over from an enclosing `with:` scope must
    continue to lower via __iadd__ (an accumulating store), not be wrongly added
    as an scf.for iter_arg by the new loop-carried collector."""
    x = torch.full((TILE, TILE), 1.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)
    expected = N_ITERS * x.float()
    _run_io_kernel(
        _make_block_aug_in_loop_kernel(),
        in_tensors=[x],
        out_zeros=[out],
        expected_list=[expected],
        dtype=dtype,
        device=device,
    )


def _make_three_acc_multi_tile_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(a_seed, b_seed, c_seed, delta, out_a, out_b, out_c):
        a_cb = ttl.make_dataflow_buffer_like(
            a_seed, shape=MULTI_TILE_SHAPE, block_count=2
        )
        b_cb = ttl.make_dataflow_buffer_like(
            b_seed, shape=MULTI_TILE_SHAPE, block_count=2
        )
        c_cb = ttl.make_dataflow_buffer_like(
            c_seed, shape=MULTI_TILE_SHAPE, block_count=2
        )
        delta_cb = ttl.make_dataflow_buffer_like(
            delta, shape=MULTI_TILE_SHAPE, block_count=N_ITERS
        )
        out_a_cb = ttl.make_dataflow_buffer_like(
            out_a, shape=MULTI_TILE_SHAPE, block_count=2
        )
        out_b_cb = ttl.make_dataflow_buffer_like(
            out_b, shape=MULTI_TILE_SHAPE, block_count=2
        )
        out_c_cb = ttl.make_dataflow_buffer_like(
            out_c, shape=MULTI_TILE_SHAPE, block_count=2
        )

        @ttl.compute()
        def compute():
            with a_cb.wait() as a, b_cb.wait() as b, c_cb.wait() as c:
                for _ in range(N_ITERS):
                    with delta_cb.wait() as d:
                        a, b, c = a + d, b + d, c + d
                with out_a_cb.reserve() as oa:
                    oa.store(a)
                with out_b_cb.reserve() as ob:
                    ob.store(b)
                with out_c_cb.reserve() as oc:
                    oc.store(c)

        @ttl.datamovement()
        def reader():
            with a_cb.reserve() as blk:
                ttl.copy(
                    a_seed[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]], blk
                ).wait()
            with b_cb.reserve() as blk:
                ttl.copy(
                    b_seed[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]], blk
                ).wait()
            with c_cb.reserve() as blk:
                ttl.copy(
                    c_seed[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]], blk
                ).wait()
            for _ in range(N_ITERS):
                with delta_cb.reserve() as blk:
                    ttl.copy(
                        delta[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]], blk
                    ).wait()

        @ttl.datamovement()
        def writer():
            with out_a_cb.wait() as blk:
                ttl.copy(
                    blk, out_a[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]]
                ).wait()
            with out_b_cb.wait() as blk:
                ttl.copy(
                    blk, out_b[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]]
                ).wait()
            with out_c_cb.wait() as blk:
                ttl.copy(
                    blk, out_c[0 : MULTI_TILE_SHAPE[0], 0 : MULTI_TILE_SHAPE[1]]
                ).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_three_accumulators_multi_tile_block(device, dtype):
    """Closes the matrix: three accumulators and multi-tile blocks together."""
    kernel = _make_three_acc_multi_tile_kernel()

    a_seed = torch.full((MULTI_TILE_ROWS, MULTI_TILE_COLS), 1.0, dtype=dtype)
    b_seed = torch.full((MULTI_TILE_ROWS, MULTI_TILE_COLS), 10.0, dtype=dtype)
    c_seed = torch.full((MULTI_TILE_ROWS, MULTI_TILE_COLS), 100.0, dtype=dtype)
    delta = torch.full((MULTI_TILE_ROWS, MULTI_TILE_COLS), 1.0, dtype=dtype)
    out_a = torch.zeros((MULTI_TILE_ROWS, MULTI_TILE_COLS), dtype=dtype)
    out_b = torch.zeros((MULTI_TILE_ROWS, MULTI_TILE_COLS), dtype=dtype)
    out_c = torch.zeros((MULTI_TILE_ROWS, MULTI_TILE_COLS), dtype=dtype)

    expected_a = a_seed.float() + N_ITERS * delta.float()
    expected_b = b_seed.float() + N_ITERS * delta.float()
    expected_c = c_seed.float() + N_ITERS * delta.float()

    a_dev = to_dram(a_seed, device)
    b_dev = to_dram(b_seed, device)
    c_dev = to_dram(c_seed, device)
    delta_dev = to_dram(delta, device)
    out_a_dev = to_dram(out_a, device)
    out_b_dev = to_dram(out_b, device)
    out_c_dev = to_dram(out_c, device)

    kernel(a_dev, b_dev, c_dev, delta_dev, out_a_dev, out_b_dev, out_c_dev)
    ttnn.synchronize_device(device)

    result_a = ttnn.to_torch(out_a_dev).float()
    result_b = ttnn.to_torch(out_b_dev).float()
    result_c = ttnn.to_torch(out_c_dev).float()
    assert_allclose(result_a, expected_a.float(), **_DTYPE_TOL[dtype])
    assert_allclose(result_b, expected_b.float(), **_DTYPE_TOL[dtype])
    assert_allclose(result_c, expected_c.float(), **_DTYPE_TOL[dtype])


def _make_zero_trip_kernel():
    @ttl.operation(grid=(1, 1))
    def kernel(initial, out):
        initial_cb = ttl.make_dataflow_buffer_like(initial, shape=(1, 1), block_count=2)
        out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            with initial_cb.wait() as state:
                # Loop bound = 0: body never executes; the iter_arg must
                # propagate from the pre-loop init store to the post-loop
                # final wait.
                for _ in range(0):
                    state = ttl.math.relu(state)
                with out_cb.reserve() as o:
                    o.store(state)

        @ttl.datamovement()
        def reader():
            with initial_cb.reserve() as blk:
                ttl.copy(initial[0:1, 0:1], blk).wait()

        @ttl.datamovement()
        def writer():
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0:1, 0:1]).wait()

    return kernel


@pytest.mark.requires_device
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_zero_trip_loop_propagates_initial_value(device, dtype):
    kernel = _make_zero_trip_kernel()

    initial = torch.full((TILE, TILE), 7.0, dtype=dtype)
    out = torch.zeros((TILE, TILE), dtype=dtype)

    expected = initial.float()

    initial_dev = to_dram(initial, device)
    out_dev = to_dram(out, device)

    kernel(initial_dev, out_dev)
    ttnn.synchronize_device(device)

    result = ttnn.to_torch(out_dev).float()
    assert_allclose(result, expected.float(), **_DTYPE_TOL[dtype])


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
def test_tuple_target_loop_carried_recurrence(device, dtype):
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
