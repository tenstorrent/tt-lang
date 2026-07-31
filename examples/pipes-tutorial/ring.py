# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_TUTORIAL_CI: requires-multi-device

"""Copy one tensor shard to the next logical device in a ring."""

from __future__ import annotations

from collections.abc import Callable

import torch
import ttl
import ttnn

from utils.correctness import assert_allclose

TILE_SIZE = 32
LOCAL_TILE_ROWS = 16
LOCAL_TILE_COLUMNS = 16
LOCAL_TENSOR_HEIGHT = LOCAL_TILE_ROWS * TILE_SIZE
LOCAL_TENSOR_WIDTH = LOCAL_TILE_COLUMNS * TILE_SIZE


def make_ring_operation(
    device_count: int,
) -> Callable[[ttnn.Tensor, ttnn.Tensor], None]:
    if device_count < 2:
        raise ValueError("a ring requires at least two logical devices")

    device_domain = ttl.DeviceDomain((device_count,))
    ring_net = ttl.PipeNet(
        graph=ttl.TransferGraph.axis_neighbor(
            device_domain,
            axis=0,
            wrap=True,
        )
    )

    @ttl.operation(grid="full", device_domain=device_domain)
    def ring_copy(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
        row_tile_count = inp.shape[0] // TILE_SIZE
        column_tile_count = inp.shape[1] // TILE_SIZE

        grid_columns, grid_rows = ttl.grid_size(dims=2)
        rows_per_node = -(-row_tile_count // grid_rows)
        columns_per_node = -(-column_tile_count // grid_columns)

        send_dfb = ttl.make_dataflow_buffer_like(
            inp,
            shape=(1, 1),
            block_count=2,
        )
        receive_dfb = ttl.make_dataflow_buffer_like(
            out,
            shape=(1, 1),
            block_count=2,
        )

        @ttl.compute()
        def idle_compute() -> None:
            pass

        @ttl.datamovement()
        def send_to_next_device() -> None:
            node_column, node_row = ttl.node(dims=2)

            for local_row in range(rows_per_node):
                row_tile = node_row * rows_per_node + local_row
                if row_tile < row_tile_count:
                    for local_column in range(columns_per_node):
                        column_tile = node_column * columns_per_node + local_column
                        if column_tile < column_tile_count:

                            def send(pipe) -> None:
                                reserved_send_block = send_dfb.reserve()
                                ttl.copy(
                                    inp[
                                        row_tile : row_tile + 1,
                                        column_tile : column_tile + 1,
                                    ],
                                    reserved_send_block,
                                ).wait()
                                reserved_send_block.push()

                                ready_send_block = send_dfb.wait()
                                ttl.copy(ready_send_block, pipe).wait()
                                ready_send_block.pop()

                            ring_net.if_src(send)

        @ttl.datamovement()
        def receive_from_previous_device() -> None:
            node_column, node_row = ttl.node(dims=2)

            for local_row in range(rows_per_node):
                row_tile = node_row * rows_per_node + local_row
                if row_tile < row_tile_count:
                    for local_column in range(columns_per_node):
                        column_tile = node_column * columns_per_node + local_column
                        if column_tile < column_tile_count:

                            def receive(pipe) -> None:
                                reserved_receive_block = receive_dfb.reserve()
                                ttl.copy(pipe, reserved_receive_block).wait()
                                reserved_receive_block.push()

                                ready_receive_block = receive_dfb.wait()
                                ttl.copy(
                                    ready_receive_block,
                                    out[
                                        row_tile : row_tile + 1,
                                        column_tile : column_tile + 1,
                                    ],
                                ).wait()
                                ready_receive_block.pop()

                            ring_net.if_dst(receive)

    return ring_copy


def _from_torch(
    tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _compose(
    tensor: ttnn.Tensor,
    mesh_device: ttnn.MeshDevice,
) -> torch.Tensor:
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


def main() -> None:
    available_device_count = ttnn.GetNumAvailableDevices()
    if available_device_count < 2:
        raise RuntimeError("this example requires at least two devices")

    device_count = min(available_device_count, 4)
    ring_copy = make_ring_operation(device_count)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape((device_count,)))

    try:
        torch.manual_seed(0)
        device_tensors = [
            torch.randn(
                (LOCAL_TENSOR_HEIGHT, LOCAL_TENSOR_WIDTH),
                dtype=torch.bfloat16,
            )
            for _ in range(device_count)
        ]
        input_torch = torch.cat(device_tensors, dim=0)
        output_torch = torch.zeros_like(input_torch)

        inp = _from_torch(input_torch, mesh_device)
        out = _from_torch(output_torch, mesh_device)
        ring_copy(inp, out)

        result = _compose(out, mesh_device)
        expected = torch.cat(
            [
                device_tensors[(destination_index - 1) % device_count]
                for destination_index in range(device_count)
            ],
            dim=0,
        )
        assert_allclose(result.float(), expected.float(), rtol=0.05, atol=1.0)
    finally:
        ttnn.close_device(mesh_device)


if __name__ == "__main__":
    main()
