# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for graph transfers between different worker coordinates."""

from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32


def _make_cross_worker_copy(mesh_shape):
    device_domain = ttl.DeviceDomain(mesh_shape)
    source_device = tuple(0 for _extent in mesh_shape)
    destination_device = tuple(extent - 1 for extent in mesh_shape)
    transfer_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(source_device, destination_device)]
        ),
        pipes=[ttl.Pipe(src=(1, 0), dst=(0, 0))],
    )

    @ttl.operation(
        grid=(2, 1),
        device_domain=device_domain,
        mesh_program_placements=[source_device, destination_device],
    )
    def cross_worker_copy(inp, out):
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

            transfer_net.if_src(send)

        @ttl.datamovement()
        def receiver():
            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()

            transfer_net.if_dst(receive)

    return cross_worker_copy


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, ttnn.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "memory_config",
    [
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram"),
        pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1"),
    ],
)
def test_graph_pipe_crosses_worker_coordinates(
    torch_dtype, ttnn_dtype, rtol, atol, memory_config
):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 2:
        pytest.skip("requires multiple devices")
    cross_worker_copy = _make_cross_worker_copy(mesh_shape)

    logical_shape = (device_count * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

        cross_worker_copy(inp, out)

        result = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(inp_torch)
    expected[-TILE_SIZE:, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
