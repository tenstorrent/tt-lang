# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""General fabric communication coverage for PipeNets."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
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
TWO_DIMENSIONAL_ROUTE_CASES = [
    pytest.param(ttnn.FabricConfig.FABRIC_2D, (0, 1), 2, id="mesh-turn"),
    pytest.param(ttnn.FabricConfig.FABRIC_2D_TORUS_X, (1,), 3, id="torus-x"),
    pytest.param(ttnn.FabricConfig.FABRIC_2D_TORUS_Y, (0,), 3, id="torus-y"),
    pytest.param(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, (0, 1), 3, id="torus-xy"),
]
ONE_DIMENSIONAL_ROUTE_CONFIGS = [
    pytest.param(ttnn.FabricConfig.FABRIC_1D, id="linear"),
    pytest.param(ttnn.FabricConfig.FABRIC_1D_RING, id="ring"),
]
# Exact physical-connectivity validation belongs to TT-Metal system-health
# tests; topology coverage here resolves routes against the live links.
RELAXED_FABRIC_INITIALIZATION = ttnn.FabricReliabilityMode.RELAXED_INIT


def requires_two_device_forwarding_link_indices(test):
    @wraps(test)
    def run_or_xfail(*args, **kwargs):
        if ttnn.get_num_devices() == 2 and not hasattr(
            ttnn, "get_forwarding_link_indices"
        ):
            pytest.xfail(
                "requires TTNN get_forwarding_link_indices() on a two-device mesh"
            )
        return test(*args, **kwargs)

    return run_or_xfail


@dataclass(frozen=True)
class FabricOperations:
    point_to_point: Callable[..., None]
    product_point_to_point: Callable[..., None]
    axis_neighbor: Callable[..., None]
    axis_neighbor_wrap: Callable[..., None]
    stencil_nearest_neighbors: Callable[..., None]
    broadcast: Callable[..., None]
    scatter: Callable[..., None]
    gather: Callable[..., None]
    reduce: Callable[..., None]
    all_reduce: Callable[..., None]
    reduce_scatter: Callable[..., None]
    all_gather: Callable[..., None]
    all_to_all: Callable[..., None]


def _make_axis_neighbor_operation(device_domain, axis_neighbor_net):
    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def exchange_axis_neighbor(inp, out):
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

    return exchange_axis_neighbor


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

    @ttl.operation(grid=(2, 1), device_domain=device_domain)
    def bidirectional_exchange(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            node_x, _node_y = ttl.node(dims=2)

            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            if node_x == 1:
                exchange_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            node_x, _node_y = ttl.node(dims=2)

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            if node_x == 1:
                exchange_net.if_dst(receive)

    return bidirectional_exchange


def _make_point_to_point_operation(
    mesh_shape: tuple[int, ...],
    source_device: tuple[int, ...],
    destination_device: tuple[int, ...],
    block_shape: tuple[int, int] = (1, 1),
):
    block_rows, block_columns = block_shape
    device_domain = ttl.DeviceDomain(mesh_shape)
    point_to_point_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(source_device, destination_device)]
        )
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def point_to_point(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0:block_rows, 0:block_columns], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            point_to_point_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0:block_rows, 0:block_columns]).wait()

            point_to_point_net.if_dst(receive)

    return point_to_point


def _flatten_device_index(coordinates, mesh_shape: tuple[int, ...]) -> int:
    device_index = 0
    for coordinate, extent in zip(coordinates, mesh_shape):
        device_index = device_index * extent + coordinate
    return device_index


def _make_high_to_low_route_endpoints(
    mesh_shape: tuple[int, ...], route_axes: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    fixed_coordinates = tuple(1 if extent > 1 else 0 for extent in mesh_shape)
    source_device = tuple(
        extent - 1 if axis in route_axes else fixed_coordinates[axis]
        for axis, extent in enumerate(mesh_shape)
    )
    destination_device = tuple(
        0 if axis in route_axes else fixed_coordinates[axis]
        for axis in range(len(mesh_shape))
    )
    return source_device, destination_device


def _select_longest_nontrivial_axis(mesh_shape: tuple[int, ...]) -> int:
    return max(
        (axis for axis, extent in enumerate(mesh_shape) if extent > 1),
        key=lambda axis: mesh_shape[axis],
    )


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
    neighbor_axis = _select_longest_nontrivial_axis(mesh_shape)
    stencil_offsets = []
    for axis, extent in enumerate(mesh_shape):
        if extent == 1:
            continue
        for delta in (-1, 1):
            offset = [0] * len(mesh_shape)
            offset[axis] = delta
            stencil_offsets.append(tuple(offset))

    point_to_point = _make_point_to_point_operation(
        mesh_shape, root_device, last_device
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
        graph=ttl.TransferGraph.axis_neighbor(device_domain, axis=neighbor_axis)
    )
    axis_neighbor_wrap_net = ttl.PipeNet(
        graph=ttl.TransferGraph.axis_neighbor(
            device_domain, axis=neighbor_axis, wrap=True
        )
    )
    axis_neighbor = _make_axis_neighbor_operation(device_domain, axis_neighbor_net)
    axis_neighbor_wrap = _make_axis_neighbor_operation(
        device_domain, axis_neighbor_wrap_net
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
        axis_neighbor_wrap=axis_neighbor_wrap,
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


def _route_mesh_options(fabric_config):
    if fabric_config != ttnn.FabricConfig.FABRIC_2D:
        return {"reliability_mode": RELAXED_FABRIC_INITIALIZATION}
    return {}


def _get_route_mesh_shape(fabric_config):
    return get_fabric_mesh_shape(
        fabric_config=fabric_config, **_route_mesh_options(fabric_config)
    )


def _open_route_mesh(mesh_shape: tuple[int, ...], fabric_config):
    return open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=fabric_config,
        **_route_mesh_options(fabric_config),
    )


@pytest.fixture(scope="module")
def fabric_mesh_shape():
    if ttnn.get_num_devices() < 2:
        pytest.skip("requires at least two devices")
    mesh_shape = _get_route_mesh_shape(ttnn.FabricConfig.FABRIC_2D)
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
        mesh.enable_program_cache()
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        program_cache_entry_counts = []
        for _ in range(20):
            fabric_operations.point_to_point(inp, out)
            program_cache_entry_counts.append(mesh.num_program_cache_entries())

        result = _compose(mesh, out)

    assert len(set(program_cache_entry_counts)) == 1
    expected = torch.zeros_like(inp_torch)
    expected[(device_count - 1) * TILE_SIZE :, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify packet splitting immediately below, at, and above the configured
# router payload limit. Completion must be published only by the final packet.
@pytest.mark.parametrize("payload_tiles", [1, 2, 3], ids=["below", "at", "above"])
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_point_to_point_packet_boundary(
    fabric_mesh_shape,
    payload_tiles,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    device_count = prod(fabric_mesh_shape)
    source_device = tuple(0 for _extent in fabric_mesh_shape)
    destination_device = tuple(extent - 1 for extent in fabric_mesh_shape)
    point_to_point = _make_point_to_point_operation(
        fabric_mesh_shape,
        source_device,
        destination_device,
        block_shape=(payload_tiles, 1),
    )
    shard_rows = payload_tiles * TILE_SIZE
    logical_shape = (device_count * shard_rows, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)
    tile_size_bytes = TILE_SIZE * TILE_SIZE * inp_torch.element_size()
    router_payload_size = 2 * tile_size_bytes
    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = router_payload_size

    with open_fabric_mesh(
        requested_mesh_shape=fabric_mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
        router_config=router_config,
    ) as mesh:
        assert ttnn.get_tt_fabric_max_payload_size_bytes() == router_payload_size
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        point_to_point(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(inp_torch)
    expected[-shard_rows:, :] = inp_torch[:shard_rows, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Receiver and sender managers serialize on one forwarding link while
# preserving the expected result and program-cache identity.
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
            mesh.enable_program_cache()
            inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
            out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

            program_cache_entry_counts = []
            for _submission in range(20):
                bidirectional_exchange(inp, out)
                program_cache_entry_counts.append(mesh.num_program_cache_entries())

            result = _compose(mesh, out)
        finally:
            ttnn.close_mesh_device(mesh)

    assert len(set(program_cache_entry_counts)) == 1
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


# Validate a mesh route that turns across axes and transfers that cross each
# configured torus boundary.
@pytest.mark.parametrize(
    "fabric_config,route_axes,minimum_extent", TWO_DIMENSIONAL_ROUTE_CASES
)
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_two_dimensional_route(
    fabric_config,
    route_axes,
    minimum_extent,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    route_mesh_shape = _get_route_mesh_shape(fabric_config)
    if len(route_mesh_shape) != 2 or any(
        route_mesh_shape[axis] < minimum_extent for axis in route_axes
    ):
        pytest.skip("requires a full 2D mesh with the requested routing topology")

    source_device, destination_device = _make_high_to_low_route_endpoints(
        route_mesh_shape, route_axes
    )
    point_to_point = _make_point_to_point_operation(
        route_mesh_shape, source_device, destination_device
    )
    device_count = prod(route_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_route_mesh(route_mesh_shape, fabric_config) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        point_to_point(inp, out)

        result = _compose(mesh, out)

    source_index = _flatten_device_index(source_device, route_mesh_shape)
    destination_index = _flatten_device_index(destination_device, route_mesh_shape)
    expected = torch.zeros_like(inp_torch)
    expected[destination_index * TILE_SIZE : (destination_index + 1) * TILE_SIZE, :] = (
        inp_torch[source_index * TILE_SIZE : (source_index + 1) * TILE_SIZE, :]
    )
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that a routing-enabled 1D fabric transfers across one logical mesh
# axis using a validated adjacent connection and explicit hop count.
@pytest.mark.parametrize("fabric_config", ONE_DIMENSIONAL_ROUTE_CONFIGS)
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_one_dimensional_route(
    fabric_config,
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    route_mesh_shape = _get_route_mesh_shape(fabric_config)
    line_axis = max(
        range(len(route_mesh_shape)), key=lambda axis: route_mesh_shape[axis]
    )
    line_shape = tuple(
        extent if axis == line_axis else 1
        for axis, extent in enumerate(route_mesh_shape)
    )
    if prod(line_shape) < 2:
        pytest.skip("requires a multi-device fabric line")

    source_device = tuple(extent - 1 for extent in line_shape)
    destination_device = tuple(0 for _extent in line_shape)
    point_to_point = _make_point_to_point_operation(
        line_shape, source_device, destination_device
    )
    device_count = prod(line_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_route_mesh(route_mesh_shape, fabric_config) as parent_mesh:
        mesh = parent_mesh.create_submesh(ttnn.MeshShape(line_shape))
        try:
            inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
            out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

            point_to_point(inp, out)

            result = _compose(mesh, out)
        finally:
            ttnn.close_mesh_device(mesh)

    expected = torch.zeros_like(inp_torch)
    expected[:TILE_SIZE, :] = inp_torch[-TILE_SIZE:, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that neighbor-exchange mode accepts an adjacent transfer without
# depending on fabric-router forwarding.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_one_dimensional_neighbor_exchange(
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    fabric_config = ttnn.FabricConfig.FABRIC_1D_NEIGHBOR_EXCHANGE
    route_mesh_shape = _get_route_mesh_shape(fabric_config)
    nontrivial_axes = tuple(
        axis for axis, extent in enumerate(route_mesh_shape) if extent > 1
    )
    if not nontrivial_axes:
        pytest.skip("requires a multi-device fabric line")
    line_axis = nontrivial_axes[0]
    line_shape = tuple(
        2 if axis == line_axis else 1 for axis in range(len(route_mesh_shape))
    )
    source_device = tuple(
        1 if axis == line_axis else 0 for axis in range(len(line_shape))
    )
    destination_device = tuple(0 for _extent in line_shape)
    point_to_point = _make_point_to_point_operation(
        line_shape, source_device, destination_device
    )
    logical_shape = (2 * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_route_mesh(route_mesh_shape, fabric_config) as parent_mesh:
        mesh = parent_mesh.create_submesh(ttnn.MeshShape(line_shape))
        try:
            inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
            out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

            point_to_point(inp, out)

            result = _compose(mesh, out)
        finally:
            ttnn.close_mesh_device(mesh)

    expected = torch.zeros_like(inp_torch)
    expected[:TILE_SIZE, :] = inp_torch[-TILE_SIZE:, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Reusing a compiled operation after reopening fabric must resolve routes for
# the active mesh and fabric configuration rather than cached prior state.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_reopen_with_different_fabric_config(
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    fabric_configs = (
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    )
    route_mesh_shapes = tuple(
        _get_route_mesh_shape(fabric_config) for fabric_config in fabric_configs
    )
    if len(set(route_mesh_shapes)) != 1:
        pytest.skip("requires one mesh extent under both fabric configurations")
    route_mesh_shape = route_mesh_shapes[0]
    if len(route_mesh_shape) != 2 or max(route_mesh_shape) < 3:
        pytest.skip("requires a full 2D mesh with a torus boundary")

    route_axis = max(
        range(len(route_mesh_shape)), key=lambda axis: route_mesh_shape[axis]
    )
    source_device, destination_device = _make_high_to_low_route_endpoints(
        route_mesh_shape, (route_axis,)
    )
    point_to_point = _make_point_to_point_operation(
        route_mesh_shape, source_device, destination_device
    )
    device_count = prod(route_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    for fabric_config in fabric_configs:
        with _open_route_mesh(route_mesh_shape, fabric_config) as mesh:
            inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
            out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

            point_to_point(inp, out)

            result = _compose(mesh, out)

        source_index = _flatten_device_index(source_device, route_mesh_shape)
        destination_index = _flatten_device_index(destination_device, route_mesh_shape)
        expected = torch.zeros_like(inp_torch)
        expected[
            destination_index * TILE_SIZE : (destination_index + 1) * TILE_SIZE,
            :,
        ] = inp_torch[
            source_index * TILE_SIZE : (source_index + 1) * TILE_SIZE,
            :,
        ]
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
    neighbor_axis = _select_longest_nontrivial_axis(fabric_mesh_shape)

    with _open_collective_mesh(fabric_mesh_shape) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        fabric_operations.axis_neighbor(inp, out)

        result = _compose(mesh, out)

    shard_shape = (*fabric_mesh_shape, TILE_SIZE, TILE_SIZE)
    expected_shards = torch.roll(
        inp_torch.reshape(shard_shape), shifts=1, dims=neighbor_axis
    )
    boundary = [slice(None)] * len(shard_shape)
    boundary[neighbor_axis] = 0
    expected_shards[tuple(boundary)] = 0
    expected = expected_shards.reshape(logical_shape)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


# Verify that a wrapped axis-neighbor relation crosses the logical boundary
# when the selected fabric configuration enables torus routing.
@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_axis_neighbor_wrap(
    torch_dtype,
    ttnn_dtype,
    rtol,
    atol,
):
    fabric_config = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    route_mesh_shape = _get_route_mesh_shape(fabric_config)
    if len(route_mesh_shape) != 2 or max(route_mesh_shape) < 3:
        pytest.skip("requires a torus axis with at least three devices")
    neighbor_axis = _select_longest_nontrivial_axis(route_mesh_shape)

    axis_neighbor_wrap = _make_fabric_operations(route_mesh_shape).axis_neighbor_wrap
    device_count = prod(route_mesh_shape)
    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with _open_route_mesh(route_mesh_shape, fabric_config) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        axis_neighbor_wrap(inp, out)

        result = _compose(mesh, out)

    shard_shape = (*route_mesh_shape, TILE_SIZE, TILE_SIZE)
    expected = torch.roll(
        inp_torch.reshape(shard_shape), shifts=1, dims=neighbor_axis
    ).reshape(logical_shape)
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
        source_index = _flatten_device_index(source, fabric_mesh_shape)
        for axis in range(len(fabric_mesh_shape)):
            for delta in (-1, 1):
                destination = list(source)
                destination[axis] += delta
                if not 0 <= destination[axis] < fabric_mesh_shape[axis]:
                    continue
                destination_index = _flatten_device_index(
                    destination, fabric_mesh_shape
                )
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
@requires_two_device_forwarding_link_indices
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


@requires_two_device_forwarding_link_indices
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


@requires_two_device_forwarding_link_indices
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
