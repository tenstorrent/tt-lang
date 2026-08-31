# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional routing-plane PipeNet ping-pong coverage."""

from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32


def _make_ping_pong_operation(mesh_shape, compiler_options):
    device_domain = ttl.DeviceDomain(mesh_shape)
    root_device = tuple(0 for _extent in mesh_shape)
    remote_device = tuple(extent - 1 for extent in mesh_shape)
    forward_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(root_device, remote_device)]
        )
    )
    return_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(remote_device, root_device)]
        )
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain, options=compiler_options)
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

            forward_net.if_src(send_forward)

            def receive_return(pipe):
                with return_dfb.reserve() as return_block:
                    ttl.copy(pipe, return_block).wait()
                with return_dfb.wait() as return_block:
                    ttl.copy(return_block, out[0, 0]).wait()

            return_net.if_dst(receive_return)

        @ttl.datamovement()
        def remote_node():
            def receive_forward(pipe):
                with forward_dfb.reserve() as forward_block:
                    ttl.copy(pipe, forward_block).wait()

            forward_net.if_dst(receive_forward)

            def send_return(pipe):
                with forward_dfb.wait() as forward_block:
                    ttl.copy(forward_block, pipe).wait()

            return_net.if_src(send_return)

    return ping_pong


@pytest.mark.parametrize(
    "torch_dtype,ttnn_dtype,rtol,atol",
    [
        pytest.param(torch.bfloat16, ttnn.bfloat16, 0.05, 1.0, id="bf16"),
        pytest.param(torch.float32, ttnn.float32, 1e-5, 1e-5, id="fp32"),
    ],
)
# Global-only mode omits local manager-ownership handoffs, so the forward and
# return intervals require one shared physical manager lifetime.
@pytest.mark.parametrize(
    "compiler_options",
    [None, "--ttl-pipe-global-semaphores-only"],
    ids=["local-first", "global-only"],
)
def test_ping_pong(torch_dtype, ttnn_dtype, rtol, atol, compiler_options):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 2:
        pytest.skip("requires multiple devices")
    ping_pong = _make_ping_pong_operation(mesh_shape, compiler_options)

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
