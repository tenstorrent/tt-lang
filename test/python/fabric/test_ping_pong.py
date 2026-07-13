# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Two-device routing-plane PipeNet ping-pong coverage."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import open_fabric_mesh
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32
DEVICE_DOMAIN = ttl.DeviceDomain((1, 4))
FORWARD_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=[((0, 0), (0, 1))])
)
RETURN_NET = ttl.PipeNet(
    graph=ttl.TransferGraph.edges(DEVICE_DOMAIN, edges=[((0, 1), (0, 0))])
)


@ttl.operation(grid=(1, 1), device_domain=DEVICE_DOMAIN)
def ping_pong(inp, out):
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    forward_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    return_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.compute()
    def idle_compute():
        pass

    @ttl.datamovement()
    def source_node():
        def send_forward(pipe):
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, 0], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        FORWARD_NET.if_src(send_forward)

        def receive_return(pipe):
            with return_dfb.reserve() as return_block:
                ttl.copy(pipe, return_block).wait()
            with return_dfb.wait() as return_block:
                ttl.copy(return_block, out[0, 0]).wait()

        RETURN_NET.if_dst(receive_return)

    @ttl.datamovement()
    def remote_node():
        def receive_forward(pipe):
            with forward_dfb.reserve() as forward_block:
                ttl.copy(pipe, forward_block).wait()

        FORWARD_NET.if_dst(receive_forward)

        def send_return(pipe):
            with forward_dfb.wait() as forward_block:
                ttl.copy(forward_block, pipe).wait()

        RETURN_NET.if_src(send_return)


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, ttnn.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
def test_ping_pong(torch_dtype, ttnn_dtype, rtol, atol):
    if ttnn.get_num_devices() < 4:
        pytest.skip("requires four devices")

    logical_shape = (4 * TILE_SIZE, TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with open_fabric_mesh(requested_mesh_shape=(1, 4)) as mesh:
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

        ping_pong(inp, out)

        result = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(inp_torch)
    expected[:TILE_SIZE, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)
