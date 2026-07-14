# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""General collective communication coverage for fabric PipeNets."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import open_fabric_mesh
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32
DEVICE_DOMAIN = ttl.DeviceDomain((1, 4))
FABRIC_DTYPES = [
    pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
    pytest.param(torch.float32, ttnn.float32, 1e-5, 1e-5, id="fp32"),
]
POINT_TO_POINT_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=[((0, 0), (0, 3))])
)
BROADCAST_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.multicast(DEVICE_DOMAIN, source=(0, 0))
)
SCATTER_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.multicast(DEVICE_DOMAIN, source=(0, 0))
)
GATHER_NET = ttl.PipeNet(graph=ttl.TransferGraph.gather(DEVICE_DOMAIN, root=(0, 0)))


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
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

        POINT_TO_POINT_NET.if_src(send)

    @ttl.datamovement()
    def receiver_node():
        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()
            with receive_dfb.wait() as receive_block:
                ttl.copy(receive_block, out[0, 0]).wait()

        POINT_TO_POINT_NET.if_dst(receive)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def broadcast(inp, out):
    local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.compute()
    def idle_compute():
        pass

    @ttl.datamovement()
    def sender_node():
        if BROADCAST_NET.is_src():
            with local_dfb.reserve() as local_block:
                ttl.copy(inp[0, 0], local_block).wait()
            with local_dfb.wait() as local_block:
                ttl.copy(local_block, out[0, 0]).wait()

        def send(pipe):
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, 0], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        BROADCAST_NET.if_src(send)

    @ttl.datamovement()
    def receiver_node():
        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()
            with receive_dfb.wait() as receive_block:
                ttl.copy(receive_block, out[0, 0]).wait()

        BROADCAST_NET.if_dst(receive)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def scatter(inp, out):
    local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.compute()
    def idle_compute():
        pass

    @ttl.datamovement()
    def sender_node():
        if SCATTER_NET.is_src():
            with local_dfb.reserve() as local_block:
                ttl.copy(inp[0, 0], local_block).wait()
            with local_dfb.wait() as local_block:
                ttl.copy(local_block, out[0, 0]).wait()

        def send(pipe):
            destination_index = pipe.destination_device.coordinates[0][1]
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[destination_index, 0], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        SCATTER_NET.if_src(send)

    @ttl.datamovement()
    def receiver_node():
        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()
            with receive_dfb.wait() as receive_block:
                ttl.copy(receive_block, out[0, 0]).wait()

        SCATTER_NET.if_dst(receive)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def gather(inp, out):
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=3)
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

        GATHER_NET.if_src(send)

    @ttl.datamovement()
    def receiver_node():
        if GATHER_NET.is_dst():
            with local_dfb.reserve() as local_block:
                ttl.copy(inp[0, 0], local_block).wait()
            with local_dfb.wait() as local_block:
                ttl.copy(local_block, out[0, 0]).wait()

        def receive(pipe):
            source_index = pipe.source_device.coordinates[0][1]
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()
            with receive_dfb.wait() as receive_block:
                ttl.copy(receive_block, out[source_index, 0]).wait()

        GATHER_NET.if_dst(receive)


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


def _require_four_devices():
    if ttnn.get_num_devices() < 4:
        pytest.skip("requires four devices")


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_point_to_point(torch_dtype, ttnn_dtype, rtol, atol):
    _require_four_devices()

    logical_shape = (4 * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with open_fabric_mesh(requested_mesh_shape=(1, 4)) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        point_to_point(inp, out)
        point_to_point(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(inp_torch)
    expected[3 * TILE_SIZE :, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_broadcast(torch_dtype, ttnn_dtype, rtol, atol):
    _require_four_devices()

    logical_shape = (4 * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with open_fabric_mesh(requested_mesh_shape=(1, 4)) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        broadcast(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch[:TILE_SIZE, :].repeat(4, 1)
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_scatter(torch_dtype, ttnn_dtype, rtol, atol):
    _require_four_devices()

    inp_shape = (16 * TILE_SIZE, TILE_SIZE)
    out_shape = (4 * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(inp_shape, dtype=torch_dtype)
    out_torch = torch.zeros(out_shape, dtype=torch_dtype)

    with open_fabric_mesh(requested_mesh_shape=(1, 4)) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        scatter(inp, out)

        result = _compose(mesh, out)

    expected = inp_torch[: 4 * TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", FABRIC_DTYPES)
def test_gather(torch_dtype, ttnn_dtype, rtol, atol):
    _require_four_devices()

    inp_shape = (4 * TILE_SIZE, TILE_SIZE)
    out_shape = (16 * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(inp_shape, dtype=torch_dtype)
    out_torch = torch.zeros(out_shape, dtype=torch_dtype)

    with open_fabric_mesh(requested_mesh_shape=(1, 4)) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, ttnn_dtype)
        out = _mesh_tensor(mesh, out_torch, ttnn_dtype)

        gather(inp, out)

        result = _compose(mesh, out)

    expected = torch.zeros_like(out_torch)
    expected[: 4 * TILE_SIZE, :] = inp_torch
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
