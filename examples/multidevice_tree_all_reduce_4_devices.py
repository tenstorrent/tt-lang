# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler
# TTLANG_TUTORIAL_CI: requires-multi-device
# type: ignore

"""Fixed four-device tree all-reduce over graph-based PipeNet syntax.

The operation launches one worker node on each logical device in a rectangular
four-device submesh. The reduce sends device 1 to 0 and 3 to 2, then 2 to 0.
The broadcast sends the final tile from 0 to 2, then from 0 to 1 and 2 to 3.
Every device writes the same reduced tile.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from math import prod

import torch
import ttl
import ttnn

from ttlang_test_utils import get_fabric_mesh_shape, open_fabric_mesh
from utils.correctness import assert_allclose

TILE_SIZE = 32
NUM_DEVICES = 4


def _device_coordinate(
    device_index: int, mesh_shape: tuple[int, ...]
) -> tuple[int, ...]:
    coordinates = []
    remaining_index = device_index
    for extent in reversed(mesh_shape):
        remaining_index, coordinate = divmod(remaining_index, extent)
        coordinates.append(coordinate)
    assert remaining_index == 0
    return tuple(reversed(coordinates))


def _select_participant_mesh_shape(
    parent_mesh_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Choose the most compact rectangular four-device submesh."""

    extent_choices = [
        tuple(
            candidate_extent
            for candidate_extent in range(1, min(parent_extent, NUM_DEVICES) + 1)
            if NUM_DEVICES % candidate_extent == 0
        )
        for parent_extent in parent_mesh_shape
    ]
    candidates = [
        candidate
        for candidate in product(*extent_choices)
        if prod(candidate) == NUM_DEVICES
    ]
    if not candidates:
        raise RuntimeError(
            f"mesh extent {parent_mesh_shape} has no rectangular four-device submesh"
        )
    return min(
        candidates,
        key=lambda candidate: (
            max(candidate),
            sum(extent * extent for extent in candidate),
            candidate,
        ),
    )


def make_tree_all_reduce_operation(
    mesh_shape: tuple[int, ...],
) -> Callable[[ttnn.Tensor, ttnn.Tensor], None]:
    mesh_shape = tuple(mesh_shape)
    if prod(mesh_shape) != NUM_DEVICES:
        raise ValueError(f"mesh_shape must contain {NUM_DEVICES} devices")
    device_domain = ttl.DeviceDomain(mesh_shape)

    def device(device_index: int):
        return _device_coordinate(device_index, mesh_shape)

    def graph(edges):
        return ttl.TransferGraph.edges(device_domain, edges=edges)

    reduce_pairs = ttl.PipeNet(
        graph=graph(
            [
                (device(1), device(0)),
                (device(3), device(2)),
            ]
        )
    )
    reduce_root = ttl.PipeNet(graph=graph([(device(2), device(0))]))
    broadcast_mid = ttl.PipeNet(graph=graph([(device(0), device(2))]))
    broadcast_leaves = ttl.PipeNet(
        graph=graph(
            [
                (device(0), device(1)),
                (device(2), device(3)),
            ]
        )
    )

    @ttl.operation(grid=(1, 1), device_domain=device_domain)
    def tree_all_reduce(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
        local_send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        local_reduce_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        pair_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        partial_send_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(1, 1), block_count=2
        )
        root_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        root_send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        mid_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        leaf_send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        leaf_recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        final_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def exchange() -> None:
            if reduce_pairs.is_src():
                with local_send_dfb.reserve() as local_blk:
                    ttl.copy(inp[0, 0], local_blk).wait()
            if reduce_pairs.is_dst():
                with local_reduce_dfb.reserve() as local_blk:
                    ttl.copy(inp[0, 0], local_blk).wait()
            if reduce_pairs.is_active():

                def send_pair(pipe) -> None:
                    with local_send_dfb.wait() as local_blk:
                        ttl.copy(local_blk, pipe).wait()

                def recv_pair(pipe) -> None:
                    with pair_recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                reduce_pairs.if_src(send_pair)
                reduce_pairs.if_dst(recv_pair)

            if reduce_root.is_active():

                def send_root(pipe) -> None:
                    with partial_send_dfb.wait() as partial_blk:
                        ttl.copy(partial_blk, pipe).wait()

                def recv_root(pipe) -> None:
                    with root_recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                reduce_root.if_src(send_root)
                reduce_root.if_dst(recv_root)

            if broadcast_mid.is_active():

                def send_mid(pipe) -> None:
                    with root_send_dfb.wait() as total_blk:
                        ttl.copy(total_blk, pipe).wait()

                def recv_mid(pipe) -> None:
                    with mid_recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                broadcast_mid.if_src(send_mid)
                broadcast_mid.if_dst(recv_mid)

            if broadcast_leaves.is_active():

                def send_leaf(pipe) -> None:
                    with leaf_send_dfb.wait() as total_blk:
                        ttl.copy(total_blk, pipe).wait()

                def recv_leaf(pipe) -> None:
                    with leaf_recv_dfb.reserve() as recv_blk:
                        ttl.copy(pipe, recv_blk).wait()

                broadcast_leaves.if_src(send_leaf)
                broadcast_leaves.if_dst(recv_leaf)

        @ttl.compute()
        def compute() -> None:
            if reduce_root.is_src():
                with (
                    local_reduce_dfb.wait() as local_blk,
                    pair_recv_dfb.wait() as remote_blk,
                    partial_send_dfb.reserve() as partial_blk,
                ):
                    partial_blk.store(local_blk + remote_blk)

            if reduce_root.is_dst():
                with (
                    local_reduce_dfb.wait() as local_blk,
                    pair_recv_dfb.wait() as pair_blk,
                    root_recv_dfb.wait() as remote_blk,
                    root_send_dfb.reserve() as root_send_blk,
                    leaf_send_dfb.reserve() as leaf_send_blk,
                    final_dfb.reserve() as final_blk,
                ):
                    total = local_blk + pair_blk + remote_blk
                    root_send_blk.store(total)
                    leaf_send_blk.store(total)
                    final_blk.store(total)

            if broadcast_mid.is_dst():
                with (
                    mid_recv_dfb.wait() as total_blk,
                    leaf_send_dfb.reserve() as leaf_send_blk,
                    final_dfb.reserve() as final_blk,
                ):
                    leaf_send_blk.store(total_blk)
                    final_blk.store(total_blk)

            if broadcast_leaves.is_dst():
                with (
                    leaf_recv_dfb.wait() as total_blk,
                    final_dfb.reserve() as final_blk,
                ):
                    final_blk.store(total_blk)

        @ttl.datamovement()
        def write_output() -> None:
            with final_dfb.wait() as final_blk:
                ttl.copy(final_blk, out[0, 0]).wait()

    return tree_all_reduce


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
        [reduced_tensor.to(torch.bfloat16) for _device_index in range(NUM_DEVICES)],
        dim=0,
    )


def main() -> None:
    if ttnn.GetNumAvailableDevices() < NUM_DEVICES:
        raise RuntimeError(f"This example requires at least {NUM_DEVICES} devices.")

    parent_mesh_shape = get_fabric_mesh_shape(fabric_config=ttnn.FabricConfig.FABRIC_2D)
    participant_mesh_shape = _select_participant_mesh_shape(parent_mesh_shape)
    tree_all_reduce = make_tree_all_reduce_operation(participant_mesh_shape)

    with open_fabric_mesh(
        requested_mesh_shape=parent_mesh_shape,
        fabric_config=ttnn.FabricConfig.FABRIC_2D,
    ) as parent_mesh:
        owns_participant_mesh = participant_mesh_shape != parent_mesh_shape
        mesh_device = (
            parent_mesh.create_submesh(ttnn.MeshShape(participant_mesh_shape))
            if owns_participant_mesh
            else parent_mesh
        )
        try:
            base_tensor = (
                torch.arange(
                    TILE_SIZE * TILE_SIZE,
                    dtype=torch.float32,
                ).reshape(TILE_SIZE, TILE_SIZE)
                / 2048.0
            )
            device_tensors = [
                (base_tensor + float(device_index + 1)).to(torch.bfloat16)
                for device_index in range(NUM_DEVICES)
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

            tree_all_reduce(input_tt, output_tt)

            result = ttnn.to_torch(
                output_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
            )
            assert_allclose(result.float(), expected.float(), rtol=5e-2, atol=1.0)

        finally:
            if owns_participant_mesh:
                ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
