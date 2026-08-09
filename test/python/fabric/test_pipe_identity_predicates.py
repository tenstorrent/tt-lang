# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Graph callback identity specialization on a Galaxy participant submesh."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32
PARENT_MESH_SHAPE = (4, 8)
PARTICIPANT_MESH_SHAPE = (1, 2)
DEVICE_DOMAIN = ttl.DeviceDomain(PARTICIPANT_MESH_SHAPE)
EXCHANGE_NET = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(DEVICE_DOMAIN))


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def identity_selected_exchange(inp, out):
    send_low_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    send_high_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    receive_low_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    receive_high_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.compute()
    def idle_compute():
        pass

    @ttl.datamovement()
    def sender_node():
        def send_to_peer(pipe):
            if pipe.destination_device_index == 0:
                with send_low_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 0], send_block).wait()
                with send_low_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()
            else:
                with send_high_dfb.reserve() as send_block:
                    ttl.copy(inp[0, 1], send_block).wait()
                with send_high_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

        EXCHANGE_NET.if_src(send_to_peer)

    @ttl.datamovement()
    def receiver_node():
        def receive_from_peer(pipe):
            source_device_index = pipe.source_device_index
            if source_device_index == 0:
                with receive_low_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_low_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 0]).wait()
            else:
                with receive_high_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_high_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, 1]).wait()

        EXCHANGE_NET.if_dst(receive_from_peer)


@pytest.fixture(scope="module")
def participant_mesh():
    required_devices = PARENT_MESH_SHAPE[0] * PARENT_MESH_SHAPE[1]
    if ttnn.GetNumAvailableDevices() < required_devices:
        pytest.skip(f"requires {required_devices} Galaxy devices")

    parent_mesh = None
    participant_submesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_X)
        parent_mesh = ttnn.open_mesh_device(ttnn.MeshShape(*PARENT_MESH_SHAPE))
        participant_submesh = parent_mesh.create_submesh(
            ttnn.MeshShape(*PARTICIPANT_MESH_SHAPE)
        )
        yield participant_submesh
    finally:
        try:
            if participant_submesh is not None:
                ttnn.close_mesh_device(participant_submesh)
        finally:
            try:
                if parent_mesh is not None:
                    ttnn.close_mesh_device(parent_mesh)
            finally:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.parametrize(
    "torch_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
def test_pipe_identity_predicates_select_exact_segments(
    participant_mesh, torch_dtype, rtol, atol
):
    logical_shape = (PARTICIPANT_MESH_SHAPE[1] * TILE_SIZE, 2 * TILE_SIZE)
    inp_torch = torch.arange(
        logical_shape[0] * logical_shape[1], dtype=torch.float32
    ).reshape(logical_shape)
    inp_torch = inp_torch.to(torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    mesh_mapper = ttnn.ShardTensorToMesh(participant_mesh, dim=0)
    inp = to_dram(inp_torch, participant_mesh, mesh_mapper=mesh_mapper)
    out = to_dram(out_torch, participant_mesh, mesh_mapper=mesh_mapper)

    identity_selected_exchange(inp, out)

    result = ttnn.to_torch(
        out,
        mesh_composer=ttnn.ConcatMeshToTensor(participant_mesh, dim=0),
    )

    expected = torch.zeros_like(out_torch)
    expected[:TILE_SIZE, TILE_SIZE:] = inp_torch[TILE_SIZE:, :TILE_SIZE]
    expected[TILE_SIZE:, :TILE_SIZE] = inp_torch[:TILE_SIZE, TILE_SIZE:]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
