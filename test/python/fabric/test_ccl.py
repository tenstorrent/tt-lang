# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""General fabric communication coverage for PipeNets."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from math import prod
import runpy

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from examples.multidevice_all_reduce import make_structured_all_reduce_operation
from ttlang_test_utils import (
    get_fabric_mesh_shape,
    open_fabric_mesh,
    requires_forwarding_link_indices,
)
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32
FABRIC_DTYPES = [
    pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
    pytest.param(torch.float32, ttnn.float32, 1e-5, 1e-5, id="fp32"),
]
# FPU inputs use TF32 precision even when the destination accumulator is FP32,
# so reduction collectives require different tolerances from data movement.
FABRIC_REDUCTION_DTYPES = [
    pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
    pytest.param(torch.float32, ttnn.float32, 5e-3, 5e-2, id="fp32"),
]


@dataclass(frozen=True)
class FabricOperations:
    point_to_point: Callable[..., None]
    product_point_to_point: Callable[..., None]
    axis_neighbor: Callable[..., None]
    stencil_nearest_neighbors: Callable[..., None]
    broadcast: Callable[..., None]
    scatter: Callable[..., None]
    gather: Callable[..., None]
    reduce: Callable[..., None]
    all_reduce: Callable[..., None]
    reduce_scatter: Callable[..., None]
    all_gather: Callable[..., None]
    all_to_all: Callable[..., None]


def _make_stencil_direction_operation(device_domain, stencil_net):
    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def exchange_direction(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            stencil_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                source_index = pipe.source_device_index
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[source_index, 0]).wait()

            stencil_net.if_dst(receive)

    return exchange_direction


def _make_bidirectional_exchange_operation(mesh_shape):
    device_domain = ttl.DeviceDomain(mesh_shape)
    root_device = tuple(0 for _extent in mesh_shape)
    peer_axis = next(axis for axis, extent in enumerate(mesh_shape) if extent > 1)
    peer_device = tuple(
        1 if axis == peer_axis else 0 for axis in range(len(mesh_shape))
    )
    exchange_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain,
            edges=[(root_device, peer_device), (peer_device, root_device)],
        )
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def bidirectional_exchange(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            exchange_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            exchange_net.if_dst(receive)

    return bidirectional_exchange


def _make_fabric_operations(
    mesh_shape: tuple[int, ...],
) -> FabricOperations:
    device_count = prod(mesh_shape)
    if device_count < 2:
        raise ValueError("fabric operations require at least two devices")

    device_domain = ttl.DeviceDomain(mesh_shape)
    root_device = tuple(0 for _extent in mesh_shape)
    last_device = tuple(extent - 1 for extent in mesh_shape)
    logical_devices = tuple(product(*(range(extent) for extent in mesh_shape)))
    # A linear cycle keeps the transfer relation O(N); target binding maps each
    # logical edge to a route supported by the selected physical topology.
    ring_edges = tuple(
        (
            logical_devices[source_index],
            logical_devices[(source_index + 1) % device_count],
        )
        for source_index in range(device_count)
    )
    receive_block_count = max(2, device_count - 1)
    ring_axis = next(axis for axis, extent in enumerate(mesh_shape) if extent > 1)
    stencil_offsets = []
    for axis, extent in enumerate(mesh_shape):
        if extent == 1:
            continue
        for delta in (-1, 1):
            offset = [0] * len(mesh_shape)
            offset[axis] = delta
            stencil_offsets.append(tuple(offset))

    point_to_point_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(device_domain, edges=[(root_device, last_device)])
    )
    product_components = {
        f"axis_{axis}": (extent,) for axis, extent in enumerate(mesh_shape)
    }
    product_domain = ttl.DeviceDomain.product(**product_components)
    product_root = ttl.DeviceRef(**{name: 0 for name in product_components})
    product_last = ttl.DeviceRef(
        **{name: extent[0] - 1 for name, extent in product_components.items()}
    )
    product_point_to_point_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            product_domain, edges=[(product_root, product_last)]
        )
    )
    axis_neighbor_net = ttl.PipeNet(
        graph=ttl.TransferGraph.axis_neighbor(device_domain, axis=ring_axis)
    )
    stencil_direction_nets = tuple(
        ttl.PipeNet(graph=ttl.TransferGraph.stencil(device_domain, offsets=[offset]))
        for offset in stencil_offsets
    )
    stencil_direction_operations = tuple(
        _make_stencil_direction_operation(device_domain, stencil_net)
        for stencil_net in stencil_direction_nets
    )
    broadcast_net = ttl.PipeNet(
        graph=ttl.TransferGraph.scatter(device_domain, source=root_device)
    )
    scatter_net = ttl.PipeNet(
        graph=ttl.TransferGraph.scatter(device_domain, source=root_device)
    )
    gather_net = ttl.PipeNet(
        graph=ttl.TransferGraph.gather(device_domain, root=root_device)
    )
    all_reduce = make_structured_all_reduce_operation(mesh_shape)
    reduce_scatter_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(device_domain, edges=ring_edges)
    )
    all_gather_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(device_domain, edges=ring_edges)
    )
    all_to_all_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(device_domain, edges=ring_edges)
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def point_to_point(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            point_to_point_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            point_to_point_net.if_dst(receive)

    @ttl.operation(grid=(1, 1), device_domain=product_domain)
    def product_point_to_point(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            product_point_to_point_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            product_point_to_point_net.if_dst(receive)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def axis_neighbor(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            axis_neighbor_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            axis_neighbor_net.if_dst(receive)

    def stencil_nearest_neighbors(inp, out):
        for exchange_direction in stencil_direction_operations:
            exchange_direction(inp, out)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def broadcast(inp, out):
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            if broadcast_net.is_src():
                with local_dfb.reserve() as local_block:
                    ttl.copy(inp[0, 0], local_block).wait()
                with local_dfb.wait() as local_block:
                    ttl.copy(local_block, out[0, 0]).wait()

            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            broadcast_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            broadcast_net.if_dst(receive)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def scatter(inp, out):
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            if scatter_net.is_src():
                with local_dfb.reserve() as local_block:
                    ttl.copy(inp[0, 0], local_block).wait()
                with local_dfb.wait() as local_block:
                    ttl.copy(local_block, out[0, 0]).wait()

            def send(pipe):
                destination_index = pipe.destination_device_index
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[destination_index, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            scatter_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            scatter_net.if_dst(receive)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def gather(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=receive_block_count
        )
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            gather_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            if gather_net.is_dst():
                with local_dfb.reserve() as local_block:
                    ttl.copy(inp[0, 0], local_block).wait()
                with local_dfb.wait() as local_block:
                    ttl.copy(local_block, out[0, 0]).wait()

            def receive(pipe):
                source_index = pipe.source_device_index
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[source_index, 0]).wait()

            gather_net.if_dst(receive)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def reduce(inp, out):
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=receive_block_count
        )
        accumulator_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        final_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            gather_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            if gather_net.is_dst():
                with local_dfb.reserve() as local_block:
                    ttl.copy(inp[0, 0], local_block).wait()

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()

            gather_net.if_dst(receive)

            if gather_net.is_dst():
                with final_dfb.wait() as final_block:
                    ttl.copy(final_block, out[0, 0]).wait()

        @ttl.compute()
        def reduce_tiles():
            if gather_net.is_dst():
                with (
                    local_dfb.wait() as local_block,
                    accumulator_dfb.reserve() as accumulator_block,
                ):
                    accumulator_block.store(local_block)

                for _remote_index in range(device_count - 2):
                    with (
                        accumulator_dfb.wait() as accumulator_block,
                        receive_dfb.wait() as remote_block,
                        accumulator_dfb.reserve() as next_accumulator_block,
                    ):
                        next_accumulator_block.store(accumulator_block + remote_block)

                with (
                    accumulator_dfb.wait() as accumulator_block,
                    receive_dfb.wait() as remote_block,
                    final_dfb.reserve() as final_block,
                ):
                    final_block.store(accumulator_block + remote_block)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def reduce_scatter(inp, out):
        initial_local_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=receive_block_count
        )
        final_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                for _round_index in range(device_count - 1):
                    with send_dfb.wait() as send_block:
                        ttl.copy(send_block, pipe).wait()

            reduce_scatter_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            device_index = device_domain.current_index()
            # Starting chunk r at device r+1 returns its fully reduced value to
            # device r after N-1 forward relays.
            initial_chunk_index = (device_index + device_count - 1) % device_count
            with initial_local_dfb.reserve() as initial_local_block:
                ttl.copy(inp[initial_chunk_index, 0], initial_local_block).wait()

            def receive(pipe):
                for round_index in range(device_count - 1):
                    local_chunk_index = (
                        device_index + device_count - round_index - 2
                    ) % device_count
                    with receive_dfb.reserve() as receive_block:
                        ttl.copy(pipe, receive_block).wait()
                    with local_dfb.reserve() as local_block:
                        ttl.copy(inp[local_chunk_index, 0], local_block).wait()

            reduce_scatter_net.if_dst(receive)

            with final_dfb.wait() as final_block:
                ttl.copy(final_block, out[0, 0]).wait()

        @ttl.compute()
        def reduce_tiles():
            with (
                initial_local_dfb.wait() as initial_local_block,
                send_dfb.reserve() as send_block,
            ):
                send_block.store(initial_local_block)

            for _round_index in range(device_count - 2):
                with (
                    receive_dfb.wait() as remote_block,
                    local_dfb.wait() as local_block,
                    send_dfb.reserve() as send_block,
                ):
                    send_block.store(remote_block + local_block)

            with (
                receive_dfb.wait() as remote_block,
                local_dfb.wait() as local_block,
                final_dfb.reserve() as final_block,
            ):
                final_block.store(remote_block + local_block)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def all_gather(inp, out):
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                for _round_index in range(device_count - 1):
                    with send_dfb.wait() as send_block:
                        ttl.copy(send_block, pipe).wait()

            all_gather_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            device_index = device_domain.current_index()
            with local_dfb.reserve() as local_block:
                ttl.copy(inp[0, 0], local_block).wait()
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, out[device_index, 0]).wait()

            def receive(pipe):
                for round_index in range(device_count - 1):
                    source_index = (
                        device_index + device_count - round_index - 1
                    ) % device_count
                    with receive_dfb.reserve() as receive_block:
                        ttl.copy(pipe, receive_block).wait()
                    with output_dfb.wait() as output_block:
                        ttl.copy(output_block, out[source_index, 0]).wait()

            all_gather_net.if_dst(receive)

        @ttl.compute()
        def relay_tiles():
            with (
                local_dfb.wait() as local_block,
                send_dfb.reserve() as send_block,
                output_dfb.reserve() as output_block,
            ):
                send_block.store(local_block)
                output_block.store(local_block)

            for _round_index in range(device_count - 2):
                with (
                    receive_dfb.wait() as receive_block,
                    output_dfb.reserve() as output_block,
                    send_dfb.reserve() as send_block,
                ):
                    output_block.store(receive_block)
                    send_block.store(receive_block)

            with (
                receive_dfb.wait() as receive_block,
                output_dfb.reserve() as output_block,
            ):
                output_block.store(receive_block)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def all_to_all(inp, out):
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.compute()
        def relay_blocks():
            for transfer_distance in range(1, device_count):
                with (
                    input_dfb.wait() as input_block,
                    send_dfb.reserve() as send_block,
                ):
                    send_block.store(input_block)

                for _relay_index in range(transfer_distance - 1):
                    with (
                        receive_dfb.wait() as receive_block,
                        send_dfb.reserve() as send_block,
                    ):
                        send_block.store(receive_block)

                with (
                    receive_dfb.wait() as receive_block,
                    output_dfb.reserve() as output_block,
                ):
                    output_block.store(receive_block)

        @ttl.datamovement()
        def sender_node():
            device_index = device_domain.current_index()

            def send(pipe):
                for destination_offset in range(1, device_count):
                    destination_index = (
                        device_index + destination_offset
                    ) % device_count
                    with input_dfb.reserve() as input_block:
                        ttl.copy(inp[destination_index, 0], input_block).wait()
                    for _block_index in range(destination_offset):
                        with send_dfb.wait() as send_block:
                            ttl.copy(send_block, pipe).wait()

            all_to_all_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            device_index = device_domain.current_index()
            with local_dfb.reserve() as local_block:
                ttl.copy(inp[device_index, 0], local_block).wait()
            with local_dfb.wait() as local_block:
                ttl.copy(local_block, out[device_index, 0]).wait()

            def receive(pipe):
                for source_offset in range(1, device_count):
                    source_index = (
                        device_index + device_count - source_offset
                    ) % device_count
                    for _block_index in range(source_offset):
                        with receive_dfb.reserve() as receive_block:
                            ttl.copy(pipe, receive_block).wait()
                    with output_dfb.wait() as output_block:
                        ttl.copy(output_block, out[source_index, 0]).wait()

            all_to_all_net.if_dst(receive)

    return FabricOperations(
        point_to_point=point_to_point,
        product_point_to_point=product_point_to_point,
        axis_neighbor=axis_neighbor,
        stencil_nearest_neighbors=stencil_nearest_neighbors,
        broadcast=broadcast,
        scatter=scatter,
        gather=gather,
        reduce=reduce,
        all_reduce=all_reduce,
        reduce_scatter=reduce_scatter,
        all_gather=all_gather,
        all_to_all=all_to_all,
    )


def _mesh_tensor(mesh, tensor, dtype):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _compose(mesh, tensor):
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )


def _reduce_source_tiles(tensor, device_count):
    tiles_per_source = tensor.shape[0] // (device_count * TILE_SIZE)
    source_tiles = tensor.reshape(device_count, tiles_per_source, TILE_SIZE, TILE_SIZE)
    return source_tiles.float().sum(dim=0).to(tensor.dtype)


def _open_collective_mesh(mesh_shape: tuple[int, ...]):
    return open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    )


@pytest.fixture(scope="module")
def fabric_mesh_shape():
    if ttnn.get_num_devices() < 2:
        pytest.skip("requires at least two devices")
    mesh_shape = get_fabric_mesh_shape()
    if prod(mesh_shape) < 2:
        pytest.skip("requires a multi-device fabric mesh")
    return mesh_shape


@pytest.fixture(scope="module")
def fabric_operations(fabric_mesh_shape):
    return _make_fabric_operations(fabric_mesh_shape)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_point_to_point(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.point_to_point(inp, out)
        fabric_operations.point_to_point(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(inp_torch)
    expected[(device_count - 1) * TILE_SIZE :, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Separate sender and receiver kernels require distinct forwarding links when
# their connections execute concurrently in the same direction.
@requires_forwarding_link_indices(ttnn)
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_bidirectional_exchange(
    fabric_mesh_shape,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    participant_axis = next(
        axis for axis, extent in enumerate(fabric_mesh_shape) if extent > 1
    )
    participant_mesh_shape = tuple(
        2 if axis == participant_axis else 1 for axis in range(len(fabric_mesh_shape))
    )
    device_count = prod(participant_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)
    bidirectional_exchange = _make_bidirectional_exchange_operation(
        participant_mesh_shape
    )

    with _open_collective_mesh(fabric_mesh_shape) as parent_mesh:
        mesh = parent_mesh.create_submesh(ttnn.MeshShape(participant_mesh_shape))
        try:
            inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
            out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

            bidirectional_exchange(inp, out)

            result = _compose(mesh, out)
        finally:
            ttnn.close_mesh_device(mesh)

    expected = torch.zeros_like(inp_torch)
    expected[:TILE_SIZE, :] = inp_torch[-TILE_SIZE:, :]
    expected[-TILE_SIZE:, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


def test_point_to_point_emitted_runner(
    fabric_mesh_shape,
    monkeypatch,
    tmp_path,
):
    """The standalone runner preserves fabric target-binding metadata."""
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(logical_shape, dtype=torch.bfloat16)
    runner_path = tmp_path / "point_to_point_runner.py"
    monkeypatch.setenv("TTLANG_EMIT_RUNNER", str(runner_path))
    # Emission occurs during first compilation; earlier tests compile the
    # module-scoped fixture operations before this environment variable is set.
    fresh_fabric_operations = _make_fabric_operations(fabric_mesh_shape)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn.bfloat16)
        compiled_out = _mesh_tensor(mesh, out_torch, ttnn.bfloat16)
        emitted_out = _mesh_tensor(mesh, out_torch, ttnn.bfloat16)

        fresh_fabric_operations.point_to_point(inp, compiled_out)
        runner = runpy.run_path(str(runner_path))
        runner["run"]([inp, emitted_out], device=mesh)
        result = _compose(mesh, emitted_out)

    expected = torch.zeros_like(inp_torch)
    expected[(device_count - 1) * TILE_SIZE :, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=0.05, atol=1.0)


# Verify that named Cartesian-product coordinates select the intended fabric
# source and destination devices.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_product_domain_point_to_point(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.product_point_to_point(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(inp_torch)
    expected[(device_count - 1) * TILE_SIZE :, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that an axis-neighbor relation transfers from each logical device to
# its successor without crossing the domain boundary.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_axis_neighbor(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)
    ring_axis = next(
        axis for axis, extent in enumerate(fabric_mesh_shape) if extent > 1
    )

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.axis_neighbor(inp, out)

        result = _compose(mesh, out)

    shard_shape = (*fabric_mesh_shape, TILE_SIZE, TILE_SIZE)
    expected_shards = torch.roll(
        inp_torch.reshape(shard_shape), shifts=1, dims=ring_axis
    )
    boundary = [slice(None)] * len(shard_shape)
    boundary[ring_axis] = 0
    expected_shards[tuple(boundary)] = 0
    expected = expected_shards.reshape(logical_shape)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that one structured relation exchanges values with nearest neighbors
# along every logical-domain axis.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_stencil_nearest_neighbors(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    inp_shape = (device_count * TILE_SIZE, TILE_SIZE)
    out_shape = (device_count * device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(inp_shape, dtype=torch_dtype)
    out_torch = torch.zeros(out_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.stencil_nearest_neighbors(inp, out)

        result = _compose(mesh, out)

    input_shards = inp_torch.reshape(device_count, TILE_SIZE, TILE_SIZE)
    expected = torch.zeros_like(out_torch).reshape(
        device_count, device_count, TILE_SIZE, TILE_SIZE
    )
    for source in product(*(range(extent) for extent in fabric_mesh_shape)):
        source_index = 0
        for coordinate, extent in zip(source, fabric_mesh_shape):
            source_index = source_index * extent + coordinate
        for axis in range(len(fabric_mesh_shape)):
            for delta in (-1, 1):
                destination = list(source)
                destination[axis] += delta
                if not 0 <= destination[axis] < fabric_mesh_shape[axis]:
                    continue
                destination_index = 0
                for coordinate, extent in zip(destination, fabric_mesh_shape):
                    destination_index = destination_index * extent + coordinate
                expected[destination_index, source_index] = input_shards[source_index]
    assert_allclose(
        result.float(),
        expected.reshape(out_shape).float(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_broadcast(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.broadcast(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch[:TILE_SIZE, :].repeat(device_count, 1)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_scatter(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    inp_shape = (device_count * device_count * TILE_SIZE, TILE_SIZE)
    out_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(inp_shape, dtype=torch_dtype)
    out_torch = torch.zeros(out_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.scatter(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch[: device_count * TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_gather(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    inp_shape = (device_count * TILE_SIZE, TILE_SIZE)
    out_shape = (device_count * device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(inp_shape, dtype=torch_dtype)
    out_torch = torch.zeros(out_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.gather(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(out_torch)
    expected[: device_count * TILE_SIZE, :] = inp_torch
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that a structured gather reduces one tile from every device and writes
# the result only at the relation root.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_REDUCTION_DTYPES)
def test_reduce(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.reduce(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(out_torch)
    expected[:TILE_SIZE, :] = _reduce_source_tiles(inp_torch, device_count)[0]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that a structured gather reduces each device tile and a structured
# scatter broadcasts the result across the discovered mesh.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_REDUCTION_DTYPES)
def test_all_reduce(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.all_reduce(inp, out)

        result = _compose(mesh, out)

    reduced = _reduce_source_tiles(inp_torch, device_count)[0]
    expected = reduced.repeat(device_count, 1)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that each destination receives and reduces its corresponding tile
# from every source device.
@requires_forwarding_link_indices(ttnn)
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_REDUCTION_DTYPES)
def test_reduce_scatter(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    inp_shape = (device_count * device_count * TILE_SIZE, TILE_SIZE)
    out_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(inp_shape, dtype=torch_dtype)
    out_torch = torch.zeros(out_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.reduce_scatter(inp, out)

        result = _compose(mesh, out)

    expected = _reduce_source_tiles(inp_torch, device_count).reshape(out_shape)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@requires_forwarding_link_indices(ttnn)
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_all_gather(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    inp_shape = (device_count * TILE_SIZE, TILE_SIZE)
    out_shape = (device_count * device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(inp_shape, dtype=torch_dtype)
    out_torch = torch.zeros(out_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.all_gather(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch.repeat(device_count, 1)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@requires_forwarding_link_indices(ttnn)
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_all_to_all(
    fabric_mesh_shape,
    fabric_operations,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    logical_shape = (device_count * device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.all_to_all(inp, out)

        result = _compose(mesh, out)

    source_destination_tiles = inp_torch.reshape(
        device_count, device_count, TILE_SIZE, TILE_SIZE
    )
    expected = source_destination_tiles.transpose(0, 1).reshape(logical_shape)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
