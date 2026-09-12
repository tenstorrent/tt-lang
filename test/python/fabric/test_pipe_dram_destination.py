# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Point-to-point PipeNet receives into receiver-owned DRAM regions."""

from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32


def _make_direct_dram_receive(mesh_shape, block_shape):
    source_device = tuple(0 for _extent in mesh_shape)
    destination_device = tuple(extent - 1 for extent in mesh_shape)
    device_domain = ttl.DeviceDomain(mesh_shape)
    transfer_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(source_device, destination_device)]
        )
    )
    block_rows, block_columns = block_shape

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def direct_dram_receive(inp, out, observed):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=block_shape, block_count=2)
        readback_dfb = ttl.make_dataflow_buffer_like(
            out, shape=block_shape, block_count=2
        )

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender_node():
            def send(pipe):
                send_block = send_dfb.reserve()
                ttl.copy(inp[0:block_rows, 0:block_columns], send_block).wait()
                ttl.copy(send_block, pipe).wait()

            transfer_net.if_src(send)

        @ttl.datamovement()
        def receiver_node():
            def receive(pipe):
                receive_request = ttl.copy(
                    pipe,
                    out[0:block_rows, 0:block_columns],
                    shape=block_shape,
                )
                receive_request.wait()

                readback_block = readback_dfb.reserve()
                ttl.copy(out[0:block_rows, 0:block_columns], readback_block).wait()
                ready_readback_block = readback_dfb.wait()
                ttl.copy(
                    ready_readback_block,
                    observed[0:block_rows, 0:block_columns],
                ).wait()

            transfer_net.if_dst(receive)

    return direct_dram_receive


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, ttnn.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
@pytest.mark.parametrize(
    "block_shape",
    [(1, 1), (2, 2), (1, 5), (2, 3)],
    ids=[
        "one-tile",
        "one-scatter-packet",
        "scatter-plus-unicast",
        "two-scatter-packets",
    ],
)
def test_pipe_receive_to_dram_region(torch_dtype, ttnn_dtype, rtol, atol, block_shape):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 2:
        pytest.skip("requires multiple devices")
    block_rows, block_columns = block_shape
    shard_shape = (block_rows * TILE_SIZE, block_columns * TILE_SIZE)
    logical_shape = (device_count * shard_shape[0], shard_shape[1])
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)
    direct_dram_receive = _make_direct_dram_receive(mesh_shape, block_shape)

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        observed = ttnn.from_torch(
            out_torch,
            dtype=ttnn_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        direct_dram_receive(inp, out, observed)

        result = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )
        observed_result = ttnn.to_torch(
            observed,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(out_torch)
    expected[-shard_shape[0] :, :] = inp_torch[: shard_shape[0], :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
    assert_allclose(observed_result.float(), expected.float(), rtol=rtol, atol=atol)
