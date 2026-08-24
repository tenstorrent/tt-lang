# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pipe protocol equivalence and computed-address schedule regressions.

The point-to-point pipe is eligible for computed receiver addresses by default;
--no-ttl-pipe-capacity-sync selects receiver-post synchronization, and
--no-ttl-pipe-computed-addresses selects receiver-published addresses.
--ttl-pipe-global-semaphores-only preserves protocol selection while placing
all synchronization counters in GlobalSemaphore storage. All configurations
must match torch and each other across dtypes. Repeated
transfers into a receiver with multiple blocks exercise sender-side slot
counters. The schedule regressions require receiver-published addresses when
control flow prevents static receiver-slot assignment, while distinct static
transfers for one PipeKey retain distinct computed addresses. Invalid schedule
regressions reject receiver address-sequence divergence and unequal rendezvous
counts.
"""

import pytest
import torch
import ttl
from ttl import ttl_api

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_pcc

TILE = 32
N_ITERS = 4
COMPLETION_ORDER_DELAY = 64


def _make_point_to_point(recv_block_count, options=None):
    @ttl.operation(grid=(2, 1), options=options)
    def point_to_point(inp, out):
        net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(
            out, shape=(1, 1), block_count=recv_block_count
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm():
            node_x, _node_y = ttl.node(dims=2)

            for _iter_idx in range(N_ITERS):

                def send(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                net.if_src(send)

                def recv(pipe):
                    with recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()
                    with recv_dfb.wait() as recv_blk:
                        ttl.copy(recv_blk, out[0, 0]).wait()

                if node_x == 1:
                    net.if_dst(recv)

        @ttl.datamovement()
        def dm_brisc():
            pass

    return point_to_point


def _make_point_to_point_ops(recv_block_count):
    return (
        _make_point_to_point(recv_block_count),
        _make_point_to_point(recv_block_count, options="--no-ttl-pipe-capacity-sync"),
        _make_point_to_point(
            recv_block_count, options="--no-ttl-pipe-computed-addresses"
        ),
        _make_point_to_point(
            recv_block_count, options="--ttl-pipe-global-semaphores-only"
        ),
    )


def _make_point_to_point_with_reset():
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    source_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    receiver_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, source_kernel, receiver_kernel),
    )

    @ttl.operation(grid=(2, 1))
    def point_to_point_with_reset(inp, out):
        net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reset_dfbs(reset, dfbs=[send_dfb])

        @ttl.datamovement(kernel=source_kernel)
        def send_data():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            net.if_src(send)
            ttl.reset_dfbs(reset, dfbs=[send_dfb])

        @ttl.datamovement(kernel=receiver_kernel)
        def receive_data():
            def receive(pipe):
                with recv_dfb.reserve() as recv_block:
                    ttl.copy(pipe, recv_block).wait()
                with recv_dfb.wait() as recv_block:
                    ttl.copy(recv_block, out[0, 0]).wait()

            net.if_dst(receive)
            ttl.reset_dfbs(reset, dfbs=[send_dfb])

    return point_to_point_with_reset


@ttl.operation(grid=(2, 1), options="--no-ttl-pipe-computed-addresses")
def loopback_collective_published_address(inp, out):
    pipe = ttl.Pipe(src=(0, 0), dst=(slice(0, 2), 0))
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)

        recv_block = recv_dfb.reserve()
        receive = ttl.copy(pipe, recv_block)

        if node_x == 0:
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, 0], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        receive.wait()
        if node_x == 0:
            ttl.copy(recv_block, out[0, 0]).wait()
        recv_block.push()

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(3, 1))
def transfer_specific_completion(inp, out):
    """Verify one transfer's completion cannot satisfy another transfer's wait."""
    multicast_pipe = ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))
    single_receiver_pipe = ttl.Pipe(src=(0, 0), dst=(slice(1, 2), 0))
    pipe_net = ttl.PipeNet([multicast_pipe, single_receiver_pipe])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    delay_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    multicast_recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)
    single_receiver_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), block_count=1
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)

        if pipe_net.is_src():
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, 1], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, single_receiver_pipe).wait()

            # Keep the first completion observable before the multicast send.
            for _delay_index in range(COMPLETION_ORDER_DELAY):
                with delay_dfb.reserve() as delay_block:
                    ttl.copy(inp[0, 1], delay_block).wait()
                with delay_dfb.wait() as _discarded_block:
                    pass

            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, 0], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, multicast_pipe).wait()

        if node_x == 1:
            multicast_block = multicast_recv_dfb.reserve()
            multicast_receive = ttl.copy(multicast_pipe, multicast_block)
            single_receiver_block = single_receiver_dfb.reserve()
            single_receiver_receive = ttl.copy(
                single_receiver_pipe, single_receiver_block
            )

            multicast_receive.wait()
            ttl.copy(multicast_block, out[0, 0]).wait()
            multicast_block.push()

            single_receiver_receive.wait()
            ttl.copy(single_receiver_block, out[0, 1]).wait()
            single_receiver_block.push()
        elif node_x == 2:
            multicast_block = multicast_recv_dfb.reserve()
            multicast_receive = ttl.copy(multicast_pipe, multicast_block)
            multicast_receive.wait()
            ttl.copy(multicast_block, out[0, 2]).wait()
            multicast_block.push()

    @ttl.datamovement()
    def dm_brisc():
        pass


# Capacity-counter, receiver-post, and receiver-published protocols, with
# local-first or global-only counter allocation, must produce identical results.
@pytest.mark.parametrize("recv_block_count", [1, 2], ids=["bc1", "bc2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_pipe_protocols_match(device, dtype, recv_block_count):
    pipe_operations = _make_point_to_point_ops(recv_block_count)
    inp_torch = torch.randn(TILE, TILE, dtype=dtype)

    configuration_results = []
    for pipe_operation in pipe_operations:
        output = to_dram(torch.zeros(TILE, TILE, dtype=dtype), device)
        pipe_operation(to_dram(inp_torch, device), output)
        ttnn.synchronize_device(device)
        configuration_results.append(ttnn.to_torch(output).float())

    for configuration_result in configuration_results:
        assert_pcc(configuration_result, inp_torch.float())
    for configuration_result in configuration_results[1:]:
        assert_pcc(configuration_results[0], configuration_result)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize(
    ("pipe_options", "expected_scratch_bytes", "expected_reset_offset"),
    [
        ("--ttl-reuse-user-dfbs", 32, 0),
        (
            "--ttl-reuse-user-dfbs --no-ttl-pipe-computed-addresses",
            64,
            32,
        ),
    ],
    ids=["computed-address", "published-address"],
)
def test_pipe_resources_coexist_with_reset(
    device,
    dtype,
    memory_config,
    to_device,
    pipe_options,
    expected_scratch_bytes,
    expected_reset_offset,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    operation = _make_point_to_point_with_reset()
    final_mlir_path = tmp_path / "pipe_with_reset.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    for invocation_index in range(2):
        input_host = (
            torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
            + invocation_index * 17
        ).to(dtype)
        input_tensor = to_device(input_host, device)
        output_tensor = to_device(torch.zeros_like(input_host), device)
        operation(input_tensor, output_tensor, options=pipe_options)
        assert_pcc(
            input_host.float(),
            ttnn.to_torch(output_tensor).float(),
        )

    final_mlir = final_mlir_path.read_text()
    assert "ttl.dfb_reset_count = 1 : i64" in final_mlir
    assert f"ttl.pipe_sram_scratch_bytes = {expected_scratch_bytes} : i64" in final_mlir
    has_computed_address_backing = "ttl.pipe_computed_address_dfb_indices" in final_mlir
    assert has_computed_address_backing == (expected_reset_offset == 0)

    compute_mlir = final_mlir.split("func.func @compute", 1)[1].split(
        "func.func @send_data", 1
    )[0]
    if expected_reset_offset == 0:
        assert "emitc.add" not in compute_mlir
    else:
        assert f"value = {expected_reset_offset} : i32" in compute_mlir
        assert "emitc.add" in compute_mlir


# A collective source that is also a receiver must publish its address through
# local L1 while the remote receiver publishes through the NoC.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_loopback_collective_published_address(device, dtype):
    inp_torch = torch.randn(TILE, TILE, dtype=dtype)
    output = to_dram(torch.zeros_like(inp_torch), device)

    loopback_collective_published_address(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    assert_pcc(inp_torch.float(), ttnn.to_torch(output).float())


# The first same-PipeNet transfer must not satisfy the delayed multicast's
# receive wait, even though both transfers share a receiver.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_receive_wait_uses_transfer_specific_completion(device, dtype):
    inp_torch = torch.randn(TILE, 2 * TILE, dtype=dtype)
    output = to_dram(torch.zeros(TILE, 3 * TILE, dtype=dtype), device)

    transfer_specific_completion(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    expected = torch.cat(
        (inp_torch[:, :TILE], inp_torch[:, TILE:], inp_torch[:, :TILE]), dim=1
    )
    assert_pcc(expected.float(), ttnn.to_torch(output).float())


@ttl.operation(grid=(3, 1))
def loop_multiplicity_pipe(inp, out):
    pipe_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(2, 0))])
    pipe_b = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(2, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for _iter_idx in range(2):

            def send_a(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            pipe_a.if_src(send_a)

            def recv_a(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            pipe_a.if_dst(recv_a)

        def send_b(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 1], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        pipe_b.if_src(send_b)

        def recv_b(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, 0]).wait()

        pipe_b.if_dst(recv_b)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(3, 1))
def branch_order_pipe(inp, out):
    pipe_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(2, 0))])
    pipe_b = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(2, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for iter_idx in range(2):
            if iter_idx > 0:

                def send_a(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                pipe_a.if_src(send_a)

                def recv_a(pipe):
                    with recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                pipe_a.if_dst(recv_a)
            else:

                def send_b(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 1], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                pipe_b.if_src(send_b)

                def recv_b(pipe):
                    with recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                pipe_b.if_dst(recv_b)

        def drain_receiver(_pipe):
            with recv_dfb.wait() as first_blk:
                ttl.copy(first_blk, out[0, 0]).wait()
            with recv_dfb.wait() as second_blk:
                ttl.copy(second_blk, out[0, 1]).wait()

        pipe_a.if_dst(drain_receiver)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def repeated_pipe_key(inp, out):
    pipe_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, 0))])
    pipe_b = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=4)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        def transfer_a_twice(pipe):
            with recv_dfb.reserve() as recv_blk:
                first_receive = ttl.copy(pipe, recv_blk)
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()
                first_receive.wait()

            with recv_dfb.reserve() as recv_blk:
                second_receive = ttl.copy(pipe, recv_blk)
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 1], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()
                second_receive.wait()

            with recv_dfb.wait() as _discarded_blk:
                pass

        pipe_a.if_dst(transfer_a_twice)

        def receive_b_then_read_oldest(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as oldest_blk:
                ttl.copy(oldest_blk, out[0, 0]).wait()

        pipe_b.if_dst(receive_b_then_read_oldest)

        def send_b(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 2], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        pipe_b.if_src(send_b)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(3, 1))
def asymmetric_multicast_receiver_traffic(inp, out):
    collective = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))])
    receiver_one = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        def advance_receiver_one(_pipe):
            with recv_dfb.reserve() as local_blk:
                ttl.copy(inp[0, 0], local_blk).wait()
            with recv_dfb.wait() as _discarded_blk:
                pass

        receiver_one.if_dst(advance_receiver_one)

        def send(pipe):
            with send_dfb.reserve() as send_blk:
                ttl.copy(inp[0, 0], send_blk).wait()
            with send_dfb.wait() as send_blk:
                ttl.copy(send_blk, pipe).wait()

        collective.if_src(send)

        def recv(pipe):
            node_x, _node_y = ttl.node(dims=2)
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as recv_blk:
                ttl.copy(recv_blk, out[0, node_x - 1]).wait()

        collective.if_dst(recv)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(3, 1))
def repeated_nonuniform_multicast(inp, out):
    collective = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0))])
    receiver_one = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for _iteration in range(2):

            def send_collective(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            collective.if_src(send_collective)

            def receive_collective(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            collective.if_dst(receive_collective)

            def send_receiver_one(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            receiver_one.if_src(send_receiver_one)

            def receive_receiver_one(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            receiver_one.if_dst(receive_receiver_one)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def mismatched_pipe_occurrences(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        def recv(pipe):
            with recv_dfb.reserve() as recv_blk:
                ttl.copy(pipe, recv_blk).wait()
            with recv_dfb.wait() as _discarded_blk:
                pass

        net.if_dst(recv)

        for _iteration in range(2):

            def send(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def node_dependent_rendezvous_count(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)
        for _iteration in range(node_x + 1):

            def send(pipe):
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                with send_dfb.wait() as send_blk:
                    ttl.copy(send_blk, pipe).wait()

            net.if_src(send)

            def receive(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            net.if_dst(receive)

    @ttl.datamovement()
    def dm_brisc():
        pass


@ttl.operation(grid=(2, 1))
def loop_conditional_rendezvous_count(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    recv_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        for iteration in range(2):
            if iteration == 0:

                def send(pipe):
                    with send_dfb.reserve() as send_blk:
                        ttl.copy(inp[0, 0], send_blk).wait()
                    with send_dfb.wait() as send_blk:
                        ttl.copy(send_blk, pipe).wait()

                net.if_src(send)

            def receive(pipe):
                with recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()
                with recv_dfb.wait() as _discarded_blk:
                    pass

            net.if_dst(receive)

    @ttl.datamovement()
    def dm_brisc():
        pass


def _random_tiles(count, dtype):
    generator = torch.Generator().manual_seed(0)
    return torch.randn((TILE, count * TILE), generator=generator).to(dtype)


# Unequal transfer multiplicities require published addresses to preserve the
# receiver's DFB order.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_loop_multiplicity_uses_published_addresses(device, dtype):
    inp_torch = _random_tiles(2, dtype)
    output = to_dram(torch.zeros((TILE, TILE), dtype=dtype), device)

    loop_multiplicity_pipe(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    assert_pcc(inp_torch[:, TILE:].float(), ttnn.to_torch(output).float())


# Mutually exclusive branches require published addresses to preserve the
# receiver's DFB order.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_branch_order_uses_published_addresses(device, dtype):
    inp_torch = _random_tiles(2, dtype)
    output = to_dram(torch.zeros_like(inp_torch), device)

    branch_order_pipe(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    expected = torch.cat((inp_torch[:, TILE:], inp_torch[:, :TILE]), dim=1)
    assert_pcc(expected.float(), ttnn.to_torch(output).float())


# Two transfer definitions for one PipeKey must use distinct computed DFB
# slots.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_repeated_pipe_key_uses_computed_addresses(device, dtype):
    inp_torch = _random_tiles(3, dtype)
    output = to_dram(torch.zeros((TILE, TILE), dtype=dtype), device)

    repeated_pipe_key(to_dram(inp_torch, device), output)
    ttnn.synchronize_device(device)

    assert_pcc(inp_torch[:, TILE : 2 * TILE].float(), ttnn.to_torch(output).float())


# Multicast must reject receivers whose prior DFB traffic produces different
# destination addresses.
def test_multicast_rejects_asymmetric_receiver_dfb_traffic(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, 2 * TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "collective pipe receiver address sequences are not proven equal "
            "for every transfer occurrence; TT-Metal NoC multicast requires "
            "one destination SRAM address for all receivers"
        ),
    ):
        asymmetric_multicast_receiver_traffic(to_dram(inp_torch, device), output)


# Multicast must reject receiver address sequences that diverge after the
# initial transfer.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_multicast_rejects_repeated_nonuniform_addresses(device, dtype):
    inp_torch = _random_tiles(1, dtype)
    inp = to_dram(inp_torch, device)
    output = to_dram(torch.zeros((TILE, 2 * TILE), dtype=dtype), device)

    with pytest.raises(
        Exception,
        match=(
            "collective pipe receiver address sequences are not proven equal "
            "for every transfer occurrence; TT-Metal NoC multicast requires "
            "one destination SRAM address for all receivers"
        ),
    ):
        repeated_nonuniform_multicast(inp, output)


# A receiver post executed once cannot synchronize with a send executed twice.
def test_pipe_rejects_different_rendezvous_execution_contexts(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "cannot prove a one-to-one synchronization schedule on PipeNet.*"
            "receiver post and send occurrences do not have matching proven "
            "execution counts and conditions"
        ),
    ):
        mismatched_pipe_occurrences(to_dram(inp_torch, device), output)


# Rendezvous counts that vary by node cannot prove a one-to-one schedule.
def test_pipe_rejects_node_dependent_rendezvous_count(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "cannot prove a one-to-one synchronization schedule on PipeNet.*"
            "receiver post and send occurrences do not have matching proven "
            "execution counts and conditions"
        ),
    ):
        node_dependent_rendezvous_count(to_dram(inp_torch, device), output)


# A conditional send cannot synchronize with an unconditional receive in each
# loop iteration.
def test_pipe_rejects_loop_conditional_rendezvous_count(device):
    inp_torch = _random_tiles(1, torch.bfloat16)
    output = to_dram(torch.zeros((TILE, TILE), dtype=torch.bfloat16), device)

    with pytest.raises(
        Exception,
        match=(
            "cannot prove a one-to-one synchronization schedule on PipeNet.*"
            "receiver post and send occurrences do not have matching proven "
            "execution counts and conditions"
        ),
    ):
        loop_conditional_rendezvous_count(to_dram(inp_torch, device), output)
