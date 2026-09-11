# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for complete Pipe endpoints containing local and fabric transfers."""

from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh, to_dram
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32


def _second_device(mesh_shape):
    coordinates = [0] * len(mesh_shape)
    for axis in range(len(mesh_shape) - 1, -1, -1):
        if mesh_shape[axis] > 1:
            coordinates[axis] = 1
            return tuple(coordinates)
    raise ValueError("complete all-to-all device test requires at least two devices")


def _make_complete_allgather(mesh_shape):
    device_domain = ttl.DeviceDomain(mesh_shape)
    first_device = tuple(0 for _extent in mesh_shape)
    second_device = _second_device(mesh_shape)
    participants = device_domain.select(
        [device_domain[first_device], device_domain[second_device]]
    )
    allgather_net = ttl.PipeNet(
        [
            ttl.Pipe.all_to_all(
                src=participants.at_node(0, 0),
                dst=participants.at_node(0, 0),
                include_self=True,
            )
        ]
    )

    @ttl.operation(
        grid=(1, 1),
        device_domain=device_domain,
        mesh_program_placements=[first_device, second_device],
    )
    def complete_allgather(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            allgather_net.if_src(send)

        @ttl.datamovement()
        def receiver():
            def receive(pipe):
                source_device_index = pipe.source_device_index
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[source_device_index, 0]).wait()

            allgather_net.if_dst(receive)

    return complete_allgather


def _mesh_tensor(mesh, tensor, memory_config):
    mesh_mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
    device_tensor = to_dram(tensor, mesh, mesh_mapper=mesh_mapper)
    if memory_config == ttnn.L1_MEMORY_CONFIG:
        device_tensor = ttnn.to_memory_config(device_tensor, memory_config)
    return device_tensor


@pytest.mark.parametrize(
    "torch_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram"),
        pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1"),
    ],
)
def test_complete_all_to_all_uses_local_and_fabric_transfers(
    torch_dtype, rtol, atol, memory_config
):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 2:
        pytest.skip("requires multiple devices")
    complete_allgather = _make_complete_allgather(mesh_shape)

    input_shape = (device_count * TILE_SIZE, TILE_SIZE)
    output_shape = (device_count * 2 * TILE_SIZE, TILE_SIZE)
    input_torch = torch.randn(input_shape, dtype=torch_dtype)
    output_torch = torch.zeros(output_shape, dtype=torch_dtype)

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh:
        inp = _mesh_tensor(mesh, input_torch, memory_config)
        out = _mesh_tensor(mesh, output_torch, memory_config)

        complete_allgather(inp, out)

        result = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(output_torch)
    source_tiles = input_torch.reshape(device_count, TILE_SIZE, TILE_SIZE)
    for destination_device_index in (0, 1):
        destination_start = destination_device_index * 2 * TILE_SIZE
        expected[destination_start : destination_start + TILE_SIZE] = source_tiles[0]
        expected[destination_start + TILE_SIZE : destination_start + 2 * TILE_SIZE] = (
            source_tiles[1]
        )
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
