# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for aggregating worker transfers through fabric forwarders."""

from math import prod

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh, to_dram
from utils.correctness import assert_allclose

pytestmark = pytest.mark.multi_device

TILE_SIZE = 32
WORKER_COUNT = 4
TRANSFER_COUNT = 2


def _make_forwarded_copy(mesh_shape):
    device_domain = ttl.DeviceDomain(mesh_shape)
    source_device = tuple(0 for _extent in mesh_shape)
    destination_device = tuple(extent - 1 for extent in mesh_shape)
    transfer_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain, edges=[(source_device, destination_device)]
        ),
        pipes=[
            ttl.Pipe(src=(worker_index, 0), dst=(worker_index, 0))
            for worker_index in range(WORKER_COUNT)
        ],
    )

    @ttl.operation(
        grid=(WORKER_COUNT, 1),
        device_domain=device_domain,
        mesh_program_placements=[source_device, destination_device],
    )
    def forwarded_copy(inp, out):
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
        receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute()
        def idle_compute():
            pass

        @ttl.datamovement()
        def sender():
            worker_index, _worker_y = ttl.node(dims=2)

            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(inp[0, worker_index], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            for _transfer_index in range(TRANSFER_COUNT):
                transfer_net.if_src(send)

        @ttl.datamovement()
        def receiver():
            worker_index, _worker_y = ttl.node(dims=2)

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                with receive_dfb.wait() as receive_block:
                    ttl.copy(receive_block, out[0, worker_index]).wait()

            for _transfer_index in range(TRANSFER_COUNT):
                transfer_net.if_dst(receive)

    return forwarded_copy


def _mesh_tensor(mesh, tensor, memory_config):
    device_tensor = to_dram(
        tensor,
        mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
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
def test_workers_share_fabric_forwarders(
    torch_dtype, rtol, atol, memory_config, monkeypatch, tmp_path
):
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    device_count = prod(mesh_shape)
    if device_count < 2:
        pytest.skip("requires multiple devices")
    forwarded_copy = _make_forwarded_copy(mesh_shape)
    final_mlir_path = tmp_path / "fabric_forwarders.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    logical_shape = (device_count * TILE_SIZE, WORKER_COUNT * TILE_SIZE)
    inp_torch = torch.randn(logical_shape, dtype=torch_dtype)
    out_torch = torch.zeros(logical_shape, dtype=torch_dtype)

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh:
        inp = _mesh_tensor(mesh, inp_torch, memory_config)
        out = _mesh_tensor(mesh, out_torch, memory_config)

        forwarded_copy(inp, out)
        forwarded_copy(inp, out)

        result = ttnn.to_torch(
            out,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        )

    expected = torch.zeros_like(inp_torch)
    expected[-TILE_SIZE:, :] = inp_torch[:TILE_SIZE, :]
    assert_allclose(result.float(), expected.float(), rtol=rtol, atol=atol)

    final_mlir = final_mlir_path.read_text()
    forwarder_nodes = "source_nodes = [array<i64: 0, 0>, array<i64: 2, 0>]"
    assert final_mlir.count(forwarder_nodes) == 2
    assert "ttl.pipe_sram_scratch_bytes" in final_mlir
