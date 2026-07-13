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
POINT_TO_POINT_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=[((0, 0), (0, 3))])
)


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


def test_point_to_point():
    if ttnn.get_num_devices() < 4:
        pytest.skip("requires four devices")

    logical_shape = (4 * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch.bfloat16)
    out_torch = torch.zeros(logical_shape, dtype=torch.bfloat16)

    with open_fabric_mesh(requested_mesh_shape=(1, 4)) as mesh:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        point_to_point(inp, out)

        result = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(inp_torch)
    expected[3 * TILE_SIZE :, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=0.05, atol=1.0)
