# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler
# TTLANG_TUTORIAL_CI: requires-multi-device
# type: ignore

"""Topology-independent all-reduce over structured transfer relations.

The operation launches one worker node on each logical device. A structured
gather reduces one tile at the root, and a structured scatter broadcasts the
result. The source program depends only on the discovered logical mesh extent;
target lowering owns the physical routes.
"""

from __future__ import annotations

from collections.abc import Callable
from math import prod

import torch
import ttl
import ttnn

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh
from utils.correctness import assert_allclose

TILE_SIZE = 32


def _validate_num_devices(num_devices: int) -> None:
    if num_devices < 2:
        raise ValueError("num_devices must be at least 2.")


def make_structured_all_reduce_operation(
    mesh_shape: tuple[int, ...],
) -> Callable[[ttnn.Tensor, ttnn.Tensor], None]:
    mesh_shape = tuple(mesh_shape)
    num_devices = prod(mesh_shape)
    _validate_num_devices(num_devices)

    device_domain = ttl.DeviceDomain(mesh_shape)
    root_device = tuple(0 for _extent in mesh_shape)
    remote_device_count = num_devices - 1
    gather_receive_block_count = max(2, remote_device_count)
    gather_net = ttl.PipeNet(
        graph=ttl.TransferGraph.gather(device_domain, root=root_device)
    )
    broadcast_net = ttl.PipeNet(
        graph=ttl.TransferGraph.scatter(device_domain, source=root_device)
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def structured_all_reduce(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
        local_root_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        gather_send_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        gather_recv_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=gather_receive_block_count
        )
        accumulator_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        broadcast_send_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        broadcast_recv_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        final_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def exchange() -> None:
            if gather_net.is_dst():
                with local_root_dfb.reserve() as local_blk:
                    ttl.copy(inp[0, 0], local_blk).wait()

            def send_gather(pipe) -> None:
                with gather_send_dfb.reserve() as local_blk:
                    ttl.copy(inp[0, 0], local_blk).wait()
                with gather_send_dfb.wait() as local_blk:
                    ttl.copy(local_blk, pipe).wait()

            def recv_gather(pipe) -> None:
                with gather_recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()

            gather_net.if_src(send_gather)
            gather_net.if_dst(recv_gather)

            def send_broadcast(pipe) -> None:
                with broadcast_send_dfb.wait() as total_blk:
                    ttl.copy(total_blk, pipe).wait()

            def recv_broadcast(pipe) -> None:
                with broadcast_recv_dfb.reserve() as recv_blk:
                    ttl.copy(pipe, recv_blk).wait()

            broadcast_net.if_src(send_broadcast)
            broadcast_net.if_dst(recv_broadcast)

        @ttl.compute()
        def compute() -> None:
            if gather_net.is_dst():
                with (
                    local_root_dfb.wait() as local_blk,
                    accumulator_dfb.reserve() as accumulator_blk,
                ):
                    accumulator_blk.store(local_blk)

                for _remote_index in range(remote_device_count - 1):
                    with (
                        accumulator_dfb.wait() as accumulator_blk,
                        gather_recv_dfb.wait() as remote_blk,
                        accumulator_dfb.reserve() as next_accumulator_blk,
                    ):
                        next_accumulator_blk.store(accumulator_blk + remote_blk)

                with (
                    accumulator_dfb.wait() as accumulator_blk,
                    gather_recv_dfb.wait() as remote_blk,
                ):
                    reduced = accumulator_blk + remote_blk
                    with final_dfb.reserve() as final_blk:
                        final_blk.store(reduced)
                    for _remote_index in range(remote_device_count):
                        with broadcast_send_dfb.reserve() as broadcast_blk:
                            broadcast_blk.store(reduced)

        @ttl.datamovement()
        def write_output() -> None:
            if gather_net.is_dst():
                with final_dfb.wait() as final_blk:
                    ttl.copy(final_blk, out[0, 0]).wait()
            else:
                with broadcast_recv_dfb.wait() as final_blk:
                    ttl.copy(final_blk, out[0, 0]).wait()

    return structured_all_reduce


def _from_torch(
    tensor: torch.Tensor,
    mesh_device,
    mesh_mapper,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _expected_reduced_tensor(device_tensors: list[torch.Tensor]) -> torch.Tensor:
    reduced_tensor = torch.zeros_like(device_tensors[0].float())
    for device_tensor in device_tensors:
        reduced_tensor = reduced_tensor + device_tensor.float()
    return torch.cat(
        [
            reduced_tensor.to(torch.bfloat16)
            for _device_index in range(len(device_tensors))
        ],
        dim=0,
    )


def main() -> None:
    mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    num_devices = prod(mesh_shape)
    all_reduce = make_structured_all_reduce_operation(mesh_shape)

    if ttnn.GetNumAvailableDevices() < num_devices:
        raise RuntimeError(f"This example requires at least {num_devices} devices.")

    with open_fabric_mesh(
        requested_mesh_shape=mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as mesh_device:
        base_tensor = (
            torch.arange(
                TILE_SIZE * TILE_SIZE,
                dtype=torch.float32,
            ).reshape(TILE_SIZE, TILE_SIZE)
            / 2048.0
        )
        device_tensors = [
            (base_tensor + float(device_index + 1)).to(torch.bfloat16)
            for device_index in range(num_devices)
        ]
        input_torch = torch.cat(device_tensors, dim=0)
        output_torch = torch.zeros_like(input_torch)
        expected = _expected_reduced_tensor(device_tensors)

        input_tt = _from_torch(
            input_torch,
            mesh_device,
            ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        output_tt = _from_torch(
            output_torch,
            mesh_device,
            ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

        all_reduce(input_tt, output_tt)

        result = ttnn.to_torch(
            output_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        assert_allclose(result.float(), expected.float(), rtol=5e-2, atol=1.0)


if __name__ == "__main__":
    main()
