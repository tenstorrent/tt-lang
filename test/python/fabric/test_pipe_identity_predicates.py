# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Graph callback identity specialization on a two-device fabric submesh."""

from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import (
    get_fabric_mesh_shape,
    open_fabric_mesh,
    requires_forwarding_link_indices,
    to_dram,
)
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32


def _make_identity_selected_exchange(participant_mesh_shape: tuple[int, ...]):
    device_domain = ttl.DeviceDomain(participant_mesh_shape)
    exchange_net = ttl.PipeNet(graph=ttl.TransferGraph.all_to_all(device_domain))

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def identity_selected_exchange(inp, out):
        send_low_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        send_high_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        receive_low_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        receive_high_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )

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

            exchange_net.if_src(send_to_peer)

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

            exchange_net.if_dst(receive_from_peer)

    return identity_selected_exchange


@pytest.fixture(scope="module")
def fabric_mesh_shape():
    if ttnn.get_num_devices() < 2:
        pytest.skip("requires at least two devices")
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    if prod(mesh_shape) < 2:
        pytest.skip("requires a multi-device fabric mesh")
    return mesh_shape


@pytest.fixture(scope="module")
def participant_mesh_shape(fabric_mesh_shape):
    participant_axis = next(
        axis for axis, extent in enumerate(fabric_mesh_shape) if extent > 1
    )
    return tuple(
        2 if axis == participant_axis else 1 for axis in range(len(fabric_mesh_shape))
    )


@pytest.fixture(scope="module")
def identity_selected_exchange(participant_mesh_shape):
    return _make_identity_selected_exchange(participant_mesh_shape)


@pytest.fixture(scope="module")
def participant_mesh(fabric_mesh_shape, participant_mesh_shape):
    with open_fabric_mesh(
        requested_mesh_shape=fabric_mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as parent_mesh:
        owns_participant_mesh = participant_mesh_shape != fabric_mesh_shape
        mesh = (
            parent_mesh.create_submesh(ttnn.MeshShape(participant_mesh_shape))
            if owns_participant_mesh
            else parent_mesh
        )
        try:
            yield mesh
        finally:
            if owns_participant_mesh:
                ttnn.close_mesh_device(mesh)


@requires_forwarding_link_indices(ttnn)
@pytest.mark.parametrize(
    "torch_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
def test_pipe_identity_predicates_select_exact_segments(
    participant_mesh_shape,
    participant_mesh,
    identity_selected_exchange,
    torch_dtype,
    rtol,
    atol,
):
    logical_shape = (prod(participant_mesh_shape) * TILE_SIZE, 2 * TILE_SIZE)
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
