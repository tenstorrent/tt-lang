# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler
# TTLANG_TUTORIAL_CI: requires-multi-device
# type: ignore

"""Parameterized full-grid tree all-reduce over planned multidevice PipeNet syntax.

This example is written against the planned API from
``/home/bnorris/tt/plans/PipesMultidevice.md``. Current ``main`` does not yet
define ``ttl.DeviceDomain``, ``ttl.TransferGraph``, or ``ttl.PipeNet(graph=...)``.

The operation launches the full core grid on each logical device. Each core
reduces its assigned local tensor tiles across matching cores on a power-of-two
device domain. The Python frontend builds the tree stages from ``num_devices``,
so user code does not hard-code the number of levels. The compiler still sees a
fixed set of PipeNets for the concrete domain used by the operation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import ttl
import ttnn

from utils.correctness import assert_allclose


TILE_SIZE = 32
NUM_DEVICES = 4
LOCAL_TILE_ROWS = 16
LOCAL_TILE_COLS = 16
LOCAL_TENSOR_HEIGHT = LOCAL_TILE_ROWS * TILE_SIZE
LOCAL_TENSOR_WIDTH = LOCAL_TILE_COLS * TILE_SIZE


def _require_multidevice_pipenet_api() -> None:
    missing = [
        name
        for name in ("DeviceDomain", "DeviceRef", "Fabric1D", "TransferGraph")
        if not hasattr(ttl, name)
    ]
    if missing:
        missing_names = ", ".join(f"ttl.{name}" for name in missing)
        raise RuntimeError(
            "This example requires planned multidevice PipeNet APIs: "
            f"{missing_names}, and ttl.PipeNet(graph=...)."
        )


def _validate_num_devices(num_devices: int) -> None:
    if num_devices < 2:
        raise ValueError("num_devices must be at least 2.")
    if num_devices & (num_devices - 1) != 0:
        raise ValueError("this binary-tree example requires power-of-two devices.")


def _make_reduce_stage_edges(num_devices: int, device):
    reduce_stage_edges = []
    distance = 1
    while distance < num_devices:
        edges = []
        for dst_column in range(0, num_devices, 2 * distance):
            src_column = dst_column + distance
            edges.append((device(src_column), device(dst_column)))
        reduce_stage_edges.append(edges)
        distance *= 2
    return reduce_stage_edges


def _reverse_edges(edges):
    return [(dst, src) for src, dst in edges]


def make_tree_all_reduce_operation(
    num_devices: int = NUM_DEVICES,
) -> Callable[[ttnn.Tensor, ttnn.Tensor], None]:
    _require_multidevice_pipenet_api()
    _validate_num_devices(num_devices)

    device_domain = ttl.DeviceDomain(
        (1, num_devices),
        topology=ttl.Fabric1D(axis=1),
    )

    def device(column: int):
        return (0, column)

    def graph(edges):
        return ttl.TransferGraph.edges(device_domain, edges=edges)

    reduce_stage_edges = _make_reduce_stage_edges(num_devices, device)
    reduce_nets = [ttl.PipeNet(graph=graph(edges)) for edges in reduce_stage_edges]
    broadcast_nets = [
        ttl.PipeNet(graph=graph(_reverse_edges(edges)))
        for edges in reversed(reduce_stage_edges)
    ]

    @ttl.operation(grid="full", device_domain=device_domain)
    def tree_all_reduce(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
        row_tiles = inp.shape[0] // TILE_SIZE
        col_tiles = inp.shape[1] // TILE_SIZE

        grid_cols, grid_rows = ttl.grid_size(dims=2)
        rows_per_core = -(-row_tiles // grid_rows)
        cols_per_core = -(-col_tiles // grid_cols)

        local_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        reduce_recv_dfbs = [
            ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            for _ in reduce_nets
        ]
        reduce_value_dfbs = [
            ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            for _ in reduce_nets
        ]
        broadcast_recv_dfbs = [
            ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            for _ in broadcast_nets
        ]
        broadcast_value_dfbs = [
            ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
            for _ in broadcast_nets
        ]
        final_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def read_local() -> None:
            core_col, core_row = ttl.node(dims=2)
            for local_row in range(rows_per_core):
                row_tile = core_row * rows_per_core + local_row
                if row_tile < row_tiles:
                    for local_col in range(cols_per_core):
                        col_tile = core_col * cols_per_core + local_col
                        if col_tile < col_tiles:
                            with local_dfb.reserve() as local_blk:
                                ttl.copy(
                                    inp[
                                        row_tile : row_tile + 1,
                                        col_tile : col_tile + 1,
                                    ],
                                    local_blk,
                                ).wait()

        @ttl.datamovement()
        def exchange() -> None:
            core_col, core_row = ttl.node(dims=2)
            for local_row in range(rows_per_core):
                row_tile = core_row * rows_per_core + local_row
                if row_tile < row_tiles:
                    for local_col in range(cols_per_core):
                        col_tile = core_col * cols_per_core + local_col
                        if col_tile < col_tiles:
                            for reduce_level, reduce_net in enumerate(reduce_nets):
                                if reduce_net.is_active():
                                    send_dfb = (
                                        local_dfb
                                        if reduce_level == 0
                                        else reduce_value_dfbs[reduce_level - 1]
                                    )
                                    recv_dfb = reduce_recv_dfbs[reduce_level]

                                    def send_reduce(pipe, send_dfb=send_dfb) -> None:
                                        with send_dfb.wait() as value_blk:
                                            ttl.copy(value_blk, pipe).wait()

                                    def recv_reduce(pipe, recv_dfb=recv_dfb) -> None:
                                        with recv_dfb.reserve() as recv_blk:
                                            ttl.copy(pipe, recv_blk).wait()

                                    reduce_net.if_src(send_reduce)
                                    reduce_net.if_dst(recv_reduce)

                            for broadcast_level, broadcast_net in enumerate(
                                broadcast_nets
                            ):
                                if broadcast_net.is_active():
                                    send_dfb = broadcast_value_dfbs[broadcast_level]
                                    recv_dfb = broadcast_recv_dfbs[broadcast_level]

                                    if broadcast_level + 1 < len(broadcast_nets):
                                        next_net = broadcast_nets[broadcast_level + 1]
                                        next_dfb = broadcast_value_dfbs[
                                            broadcast_level + 1
                                        ]

                                        def send_broadcast(
                                            pipe,
                                            send_dfb=send_dfb,
                                            next_net=next_net,
                                            next_dfb=next_dfb,
                                        ) -> None:
                                            with send_dfb.wait() as value_blk:
                                                ttl.copy(value_blk, pipe).wait()
                                                if next_net.is_src():
                                                    with next_dfb.reserve() as next_blk:
                                                        next_blk.store(value_blk)

                                    else:

                                        def send_broadcast(
                                            pipe,
                                            send_dfb=send_dfb,
                                        ) -> None:
                                            with send_dfb.wait() as value_blk:
                                                ttl.copy(value_blk, pipe).wait()

                                    def recv_broadcast(pipe, recv_dfb=recv_dfb) -> None:
                                        with recv_dfb.reserve() as recv_blk:
                                            ttl.copy(pipe, recv_blk).wait()

                                    broadcast_net.if_src(send_broadcast)
                                    broadcast_net.if_dst(recv_broadcast)

        @ttl.compute()
        def compute() -> None:
            core_col, core_row = ttl.node(dims=2)
            for local_row in range(rows_per_core):
                row_tile = core_row * rows_per_core + local_row
                if row_tile < row_tiles:
                    for local_col in range(cols_per_core):
                        col_tile = core_col * cols_per_core + local_col
                        if col_tile < col_tiles:
                            for reduce_level, reduce_net in enumerate(reduce_nets):
                                if reduce_net.is_dst():
                                    input_dfb = (
                                        local_dfb
                                        if reduce_level == 0
                                        else reduce_value_dfbs[reduce_level - 1]
                                    )
                                    recv_dfb = reduce_recv_dfbs[reduce_level]
                                    value_dfb = reduce_value_dfbs[reduce_level]

                                    if reduce_level == len(reduce_nets) - 1:
                                        with (
                                            input_dfb.wait() as local_blk,
                                            recv_dfb.wait() as remote_blk,
                                            value_dfb.reserve() as value_blk,
                                            broadcast_value_dfbs[
                                                0
                                            ].reserve() as broadcast_blk,
                                            final_dfb.reserve() as final_blk,
                                        ):
                                            total = local_blk + remote_blk
                                            value_blk.store(total)
                                            broadcast_blk.store(total)
                                            final_blk.store(total)
                                    else:
                                        with (
                                            input_dfb.wait() as local_blk,
                                            recv_dfb.wait() as remote_blk,
                                            value_dfb.reserve() as value_blk,
                                        ):
                                            value_blk.store(local_blk + remote_blk)

                            for broadcast_level, broadcast_net in enumerate(
                                broadcast_nets
                            ):
                                if broadcast_net.is_dst():
                                    recv_dfb = broadcast_recv_dfbs[broadcast_level]
                                    if broadcast_level + 1 < len(broadcast_nets):
                                        next_net = broadcast_nets[broadcast_level + 1]
                                        if next_net.is_src():
                                            next_dfb = broadcast_value_dfbs[
                                                broadcast_level + 1
                                            ]
                                            with (
                                                recv_dfb.wait() as total_blk,
                                                next_dfb.reserve() as next_blk,
                                                final_dfb.reserve() as final_blk,
                                            ):
                                                next_blk.store(total_blk)
                                                final_blk.store(total_blk)
                                        else:
                                            with (
                                                recv_dfb.wait() as total_blk,
                                                final_dfb.reserve() as final_blk,
                                            ):
                                                final_blk.store(total_blk)
                                    else:
                                        with (
                                            recv_dfb.wait() as total_blk,
                                            final_dfb.reserve() as final_blk,
                                        ):
                                            final_blk.store(total_blk)

        @ttl.datamovement()
        def write_output() -> None:
            core_col, core_row = ttl.node(dims=2)
            for local_row in range(rows_per_core):
                row_tile = core_row * rows_per_core + local_row
                if row_tile < row_tiles:
                    for local_col in range(cols_per_core):
                        col_tile = core_col * cols_per_core + local_col
                        if col_tile < col_tiles:
                            with final_dfb.wait() as final_blk:
                                ttl.copy(
                                    final_blk,
                                    out[
                                        row_tile : row_tile + 1,
                                        col_tile : col_tile + 1,
                                    ],
                                ).wait()

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
        [reduced_tensor.to(torch.bfloat16) for _ in range(len(device_tensors))],
        dim=0,
    )


def main() -> None:
    num_devices = NUM_DEVICES
    tree_all_reduce = make_tree_all_reduce_operation(num_devices)

    if ttnn.GetNumAvailableDevices() < num_devices:
        raise RuntimeError(f"This example requires at least {num_devices} devices.")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, num_devices))

    try:
        base_tensor = (
            torch.arange(
                LOCAL_TENSOR_HEIGHT * LOCAL_TENSOR_WIDTH,
                dtype=torch.float32,
            ).reshape(LOCAL_TENSOR_HEIGHT, LOCAL_TENSOR_WIDTH)
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

        tree_all_reduce(input_tt, output_tt)

        result = ttnn.to_torch(
            output_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
        )
        assert_allclose(result.float(), expected.float(), rtol=5e-2, atol=1.0)

    finally:
        ttnn.close_device(mesh_device)


if __name__ == "__main__":
    main()
