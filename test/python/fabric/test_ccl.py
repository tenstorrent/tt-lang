# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""General collective communication coverage for fabric PipeNets."""

from collections.abc import Callable
from dataclasses import dataclass
from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32
FABRIC_DTYPES = [
    pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
    pytest.param(torch.float32, ttnn.float32, 1e-5, 1e-5, id="fp32"),
]


@dataclass(frozen=True)
class CollectiveOperations:
    point_to_point: Callable[..., None]
    broadcast: Callable[..., None]
    scatter: Callable[..., None]
    gather: Callable[..., None]
    all_gather: Callable[..., None]
    all_to_all: Callable[..., None]


def _make_collective_operations(
    mesh_shape: tuple[int, ...],
) -> CollectiveOperations:
    device_count = prod(mesh_shape)
    if device_count < 2:
        raise ValueError("collective operations require at least two devices")

    device_domain = ttl.DeviceDomain(mesh_shape)
    root_device = tuple(0 for _ in mesh_shape)
    last_device = tuple(extent - 1 for extent in mesh_shape)
    receive_block_count = max(2, device_count - 1)

    point_to_point_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(device_domain, edges=[(root_device, last_device)])
    )
    broadcast_net = ttl.PipeNet(
        graph=ttl.TransferGraph.multicast(device_domain, source=root_device)
    )
    scatter_net = ttl.PipeNet(
        graph=ttl.TransferGraph.multicast(device_domain, source=root_device)
    )
    gather_net = ttl.PipeNet(
        graph=ttl.TransferGraph.gather(device_domain, root=root_device)
    )
    all_gather_net = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(device_domain))
    all_to_all_net = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(device_domain))

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
    def all_gather(inp, out):
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=receive_block_count
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            device_index = device_domain.current_index()
            with local_dfb.reserve() as local_block:
                ttl.copy(inp[0, 0], local_block).wait()
            with local_dfb.wait() as local_block:
                ttl.copy(local_block, out[device_index, 0]).wait()

            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            all_gather_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                source_index = pipe.source_device_index
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[source_index, 0]).wait()

            all_gather_net.if_dst(receive)

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def all_to_all(inp, out):
        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=receive_block_count
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            device_index = device_domain.current_index()
            with local_dfb.reserve() as local_block:
                ttl.copy(inp[device_index, 0], local_block).wait()
            with local_dfb.wait() as local_block:
                ttl.copy(local_block, out[device_index, 0]).wait()

            def send(pipe):
                destination_index = pipe.destination_device_index
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[destination_index, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            all_to_all_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                source_index = pipe.source_device_index
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[source_index, 0]).wait()

            all_to_all_net.if_dst(receive)

    return CollectiveOperations(
        point_to_point=point_to_point,
        broadcast=broadcast,
        scatter=scatter,
        gather=gather,
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
def collective_operations(fabric_mesh_shape):
    return _make_collective_operations(fabric_mesh_shape)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_point_to_point(
    fabric_mesh_shape,
    collective_operations,
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

        collective_operations.point_to_point(inp, out)
        collective_operations.point_to_point(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(inp_torch)
    expected[(device_count - 1) * TILE_SIZE :, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_broadcast(
    fabric_mesh_shape,
    collective_operations,
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

        collective_operations.broadcast(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch[:TILE_SIZE, :].repeat(device_count, 1)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_scatter(
    fabric_mesh_shape,
    collective_operations,
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

        collective_operations.scatter(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch[: device_count * TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_gather(
    fabric_mesh_shape,
    collective_operations,
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

        collective_operations.gather(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(out_torch)
    expected[: device_count * TILE_SIZE, :] = inp_torch
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_all_gather(
    fabric_mesh_shape,
    collective_operations,
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

        collective_operations.all_gather(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch.repeat(device_count, 1)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_all_to_all(
    fabric_mesh_shape,
    collective_operations,
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

        collective_operations.all_to_all(inp, out)

        result = _compose(mesh, out)

    source_destination_tiles = inp_torch.reshape(
        device_count, device_count, TILE_SIZE, TILE_SIZE
    )
    expected = source_destination_tiles.transpose(0, 1).reshape(logical_shape)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
