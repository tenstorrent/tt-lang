# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_TUTORIAL_CI: requires-multi-device

"""Copy tensor shards through row and column rings in a logical device grid."""

from __future__ import annotations

import torch
import ttl
import ttnn

from utils.correctness import assert_allclose

TILE_SIZE = 32
ROW_COUNT = 2
COLUMN_COUNT = 2
DEVICE_COUNT = ROW_COUNT * COLUMN_COUNT
LOCAL_TILE_ROWS = 16
LOCAL_TILE_COLUMNS = 16
LOCAL_TENSOR_HEIGHT = LOCAL_TILE_ROWS * TILE_SIZE
LOCAL_TENSOR_WIDTH = LOCAL_TILE_COLUMNS * TILE_SIZE

device_domain = ttl.DeviceDomain((ROW_COUNT, COLUMN_COUNT))
row_ring_net = ttl.PipeNet(
    graph=ttl.TransferGraph.axis_neighbor(
        device_domain,
        axis=1,
        wrap=True,
    )
)
column_ring_net = ttl.PipeNet(
    graph=ttl.TransferGraph.axis_neighbor(
        device_domain,
        axis=0,
        wrap=True,
    )
)


@ttl.operation(grid="full", device_domain=device_domain)
def row_and_column_ring_copy(
    inp: ttnn.Tensor,
    row_out: ttnn.Tensor,
    column_out: ttnn.Tensor,
) -> None:
    row_tile_count = inp.shape[0] // TILE_SIZE
    column_tile_count = inp.shape[1] // TILE_SIZE

    grid_columns, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tile_count // grid_rows)
    columns_per_node = -(-column_tile_count // grid_columns)

    row_send_dfb = ttl.make_dataflow_buffer_like(
        inp,
        shape=(1, 1),
        block_count=2,
    )
    row_receive_dfb = ttl.make_dataflow_buffer_like(
        row_out,
        shape=(1, 1),
        block_count=2,
    )
    column_send_dfb = ttl.make_dataflow_buffer_like(
        inp,
        shape=(1, 1),
        block_count=2,
    )
    column_receive_dfb = ttl.make_dataflow_buffer_like(
        column_out,
        shape=(1, 1),
        block_count=2,
    )

    @ttl.compute()
    def idle_compute() -> None:
        pass

    @ttl.datamovement()
    def send_to_next_devices() -> None:
        node_column, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):
            row_tile = node_row * rows_per_node + local_row
            if row_tile < row_tile_count:
                for local_column in range(columns_per_node):
                    column_tile = node_column * columns_per_node + local_column
                    if column_tile < column_tile_count:

                        def send_row(pipe) -> None:
                            reserved_send_block = row_send_dfb.reserve()
                            ttl.copy(
                                inp[
                                    row_tile : row_tile + 1,
                                    column_tile : column_tile + 1,
                                ],
                                reserved_send_block,
                            ).wait()
                            reserved_send_block.push()

                            ready_send_block = row_send_dfb.wait()
                            ttl.copy(ready_send_block, pipe).wait()
                            ready_send_block.pop()

                        row_ring_net.if_src(send_row)

                        def send_column(pipe) -> None:
                            reserved_send_block = column_send_dfb.reserve()
                            ttl.copy(
                                inp[
                                    row_tile : row_tile + 1,
                                    column_tile : column_tile + 1,
                                ],
                                reserved_send_block,
                            ).wait()
                            reserved_send_block.push()

                            ready_send_block = column_send_dfb.wait()
                            ttl.copy(ready_send_block, pipe).wait()
                            ready_send_block.pop()

                        column_ring_net.if_src(send_column)

    @ttl.datamovement()
    def receive_from_previous_devices() -> None:
        node_column, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):
            row_tile = node_row * rows_per_node + local_row
            if row_tile < row_tile_count:
                for local_column in range(columns_per_node):
                    column_tile = node_column * columns_per_node + local_column
                    if column_tile < column_tile_count:

                        def receive_row(pipe) -> None:
                            reserved_receive_block = row_receive_dfb.reserve()
                            ttl.copy(pipe, reserved_receive_block).wait()
                            reserved_receive_block.push()

                            ready_receive_block = row_receive_dfb.wait()
                            ttl.copy(
                                ready_receive_block,
                                row_out[
                                    row_tile : row_tile + 1,
                                    column_tile : column_tile + 1,
                                ],
                            ).wait()
                            ready_receive_block.pop()

                        row_ring_net.if_dst(receive_row)

                        def receive_column(pipe) -> None:
                            reserved_receive_block = column_receive_dfb.reserve()
                            ttl.copy(pipe, reserved_receive_block).wait()
                            reserved_receive_block.push()

                            ready_receive_block = column_receive_dfb.wait()
                            ttl.copy(
                                ready_receive_block,
                                column_out[
                                    row_tile : row_tile + 1,
                                    column_tile : column_tile + 1,
                                ],
                            ).wait()
                            ready_receive_block.pop()

                        column_ring_net.if_dst(receive_column)


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


def _expected_row_ring(device_tensors: list[torch.Tensor]) -> torch.Tensor:
    source_tensors = []
    for destination_row in range(ROW_COUNT):
        for destination_column in range(COLUMN_COUNT):
            source_column = (destination_column - 1) % COLUMN_COUNT
            source_index = destination_row * COLUMN_COUNT + source_column
            source_tensors.append(device_tensors[source_index])
    return torch.cat(source_tensors, dim=0)


def _expected_column_ring(device_tensors: list[torch.Tensor]) -> torch.Tensor:
    source_tensors = []
    for destination_row in range(ROW_COUNT):
        for destination_column in range(COLUMN_COUNT):
            source_row = (destination_row - 1) % ROW_COUNT
            source_index = source_row * COLUMN_COUNT + destination_column
            source_tensors.append(device_tensors[source_index])
    return torch.cat(source_tensors, dim=0)


def main() -> None:
    if ttnn.GetNumAvailableDevices() < DEVICE_COUNT:
        raise RuntimeError(f"this example requires at least {DEVICE_COUNT} devices")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(ROW_COUNT, COLUMN_COUNT))

    try:
        torch.manual_seed(0)
        device_tensors = [
            torch.randn(
                (LOCAL_TENSOR_HEIGHT, LOCAL_TENSOR_WIDTH),
                dtype=torch.bfloat16,
            )
            for _ in range(DEVICE_COUNT)
        ]
        input_torch = torch.cat(device_tensors, dim=0)
        output_torch = torch.zeros_like(input_torch)

        inp = _from_torch(input_torch, mesh_device)
        row_out = _from_torch(output_torch, mesh_device)
        column_out = _from_torch(output_torch, mesh_device)
        row_and_column_ring_copy(inp, row_out, column_out)

        row_result = _compose(row_out, mesh_device)
        column_result = _compose(column_out, mesh_device)
        assert_allclose(
            row_result.float(),
            _expected_row_ring(device_tensors).float(),
            rtol=0.05,
            atol=1.0,
        )
        assert_allclose(
            column_result.float(),
            _expected_column_ring(device_tensors).float(),
            rtol=0.05,
            atol=1.0,
        )
    finally:
        ttnn.close_device(mesh_device)


if __name__ == "__main__":
    main()
