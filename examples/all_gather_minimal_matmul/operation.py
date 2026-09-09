# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fabric all-gather fused with a distributed block matmul and bias."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod

import ttl
import ttnn

MAX_DEVICE_COUNT = 32


@dataclass(frozen=True)
class AllGatherMinimalMatmulConfig:
    """Static tile and device decomposition for the fused operation."""

    mesh_shape: tuple[int, ...]
    m_tiles: int
    k_tiles_per_device: int
    n_tiles_per_device: int
    m_block_tiles: int = 1
    k_tiles_per_transfer: int = 1
    n_block_tiles: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh_shape", tuple(self.mesh_shape))
        if not self.mesh_shape or any(extent <= 0 for extent in self.mesh_shape):
            raise ValueError("mesh_shape must contain positive extents")
        if self.device_count < 2:
            raise ValueError("all-gather requires at least two devices")
        if self.device_count > MAX_DEVICE_COUNT:
            raise ValueError(f"all-gather supports at most {MAX_DEVICE_COUNT} devices")

        tile_fields = {
            "m_tiles": self.m_tiles,
            "k_tiles_per_device": self.k_tiles_per_device,
            "n_tiles_per_device": self.n_tiles_per_device,
            "m_block_tiles": self.m_block_tiles,
            "k_tiles_per_transfer": self.k_tiles_per_transfer,
            "n_block_tiles": self.n_block_tiles,
        }
        for field_name, value in tile_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        divisibility_constraints = (
            ("m_tiles", self.m_tiles, "m_block_tiles", self.m_block_tiles),
            (
                "k_tiles_per_device",
                self.k_tiles_per_device,
                "k_tiles_per_transfer",
                self.k_tiles_per_transfer,
            ),
            (
                "n_tiles_per_device",
                self.n_tiles_per_device,
                "n_block_tiles",
                self.n_block_tiles,
            ),
        )
        for total_name, total, block_name, block in divisibility_constraints:
            if total % block != 0:
                raise ValueError(f"{total_name} must be divisible by {block_name}")

        grid_columns, grid_rows = self.grid
        if grid_columns < 2 or grid_rows < 2:
            raise ValueError(
                "full-grid scheduling requires at least two M and two N workers"
            )

    @property
    def device_count(self) -> int:
        return prod(self.mesh_shape)

    @property
    def grid(self) -> tuple[int, int]:
        return (
            self.n_tiles_per_device // self.n_block_tiles,
            self.m_tiles // self.m_block_tiles,
        )

    @property
    def k_transfer_count(self) -> int:
        return self.k_tiles_per_device // self.k_tiles_per_transfer


def make_all_gather_minimal_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
) -> Callable[..., None]:
    """Create ``output = all_gather(activation) @ weight + bias``.

    Activations are K-sharded and weights, bias, and output are N-sharded over
    ``mesh_shape``. One worker column performs each activation fabric transfer;
    local row broadcasts distribute the gathered blocks across N workers.
    Weight column broadcasts distribute each N block across M workers.

    Inputs and outputs use TILE layout in DRAM. A zero bias tensor selects the
    unbiased computation without requiring a second compiled operation.
    """

    device_domain = ttl.DeviceDomain(config.mesh_shape)
    activation_all_gather_net = ttl.PipeNet(
        graph=ttl.TransferGraph.all_to_all(device_domain)
    )

    grid_columns, grid_rows = config.grid
    activation_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, row_index),
                dst=(slice(1, grid_columns), row_index),
            )
            for row_index in range(grid_rows)
        ]
    )
    weight_column_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(column_index, 0),
                dst=(column_index, slice(1, grid_rows)),
            )
            for column_index in range(grid_columns)
        ]
    )

    m_block_tiles = config.m_block_tiles
    k_tiles_per_device = config.k_tiles_per_device
    k_tiles_per_transfer = config.k_tiles_per_transfer
    n_block_tiles = config.n_block_tiles
    device_count = config.device_count
    k_transfer_count = config.k_transfer_count
    receive_block_count = max(2, device_count - 1)

    @ttl.operation(
        grid=config.grid,
        device_domain=device_domain,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    def all_gather_minimal_matmul(
        activation_shard: ttnn.Tensor,
        weight_shard: ttnn.Tensor,
        bias_shard: ttnn.Tensor,
        gathered_activation: ttnn.Tensor,
        output_shard: ttnn.Tensor,
    ) -> None:
        fabric_send_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, k_tiles_per_transfer),
            block_count=2,
        )
        local_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, k_tiles_per_transfer),
            block_count=2,
        )
        fabric_receive_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, k_tiles_per_transfer),
            block_count=receive_block_count,
        )
        activation_compute_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, k_tiles_per_transfer),
            block_count=2,
        )
        weight_compute_dfb = ttl.make_dataflow_buffer_like(
            weight_shard,
            shape=(k_tiles_per_transfer, n_block_tiles),
            block_count=2,
        )
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias_shard, shape=(1, n_block_tiles), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output_shard,
            shape=(m_block_tiles, n_block_tiles),
            block_count=2,
        )

        @ttl.datamovement()
        def send_activation_and_distribute_weights() -> None:
            node_column, node_row = ttl.node(dims=2)
            m_begin = node_row * m_block_tiles
            m_end = m_begin + m_block_tiles
            n_begin = node_column * n_block_tiles
            n_end = n_begin + n_block_tiles

            if node_column == 0:

                def send_activation_to_device(pipe) -> None:
                    for transfer_index in range(k_transfer_count):
                        local_k_begin = transfer_index * k_tiles_per_transfer
                        local_k_end = local_k_begin + k_tiles_per_transfer
                        with fabric_send_dfb.reserve() as activation_block:
                            ttl.copy(
                                activation_shard[
                                    m_begin:m_end, local_k_begin:local_k_end
                                ],
                                activation_block,
                            ).wait()
                        with fabric_send_dfb.wait() as activation_block:
                            ttl.copy(activation_block, pipe).wait()

                activation_all_gather_net.if_src(send_activation_to_device)

            for source_index in range(device_count):
                source_k_begin = source_index * k_tiles_per_device
                for transfer_index in range(k_transfer_count):
                    k_begin = source_k_begin + transfer_index * k_tiles_per_transfer
                    k_end = k_begin + k_tiles_per_transfer
                    with weight_compute_dfb.reserve() as weight_block:
                        if node_row == 0:
                            ttl.copy(
                                weight_shard[k_begin:k_end, n_begin:n_end],
                                weight_block,
                            ).wait()

                            def send_weight(pipe) -> None:
                                ttl.copy(weight_block, pipe).wait()

                            weight_column_net.if_src(send_weight)
                        else:

                            def receive_weight(pipe) -> None:
                                ttl.copy(pipe, weight_block).wait()

                            weight_column_net.if_dst(receive_weight)

            with bias_dfb.reserve() as bias_block:
                ttl.copy(bias_shard[0:1, n_begin:n_end], bias_block).wait()

        @ttl.datamovement()
        def receive_activation_and_write_output() -> None:
            node_column, node_row = ttl.node(dims=2)
            m_begin = node_row * m_block_tiles
            m_end = m_begin + m_block_tiles

            if node_column == 0:
                local_device_index = device_domain.current_index()
                local_device_k_begin = local_device_index * k_tiles_per_device

                for transfer_index in range(k_transfer_count):
                    local_k_begin = transfer_index * k_tiles_per_transfer
                    local_k_end = local_k_begin + k_tiles_per_transfer
                    gathered_k_begin = local_device_k_begin + local_k_begin
                    gathered_k_end = gathered_k_begin + k_tiles_per_transfer
                    with local_activation_dfb.reserve() as activation_block:
                        ttl.copy(
                            activation_shard[m_begin:m_end, local_k_begin:local_k_end],
                            activation_block,
                        ).wait()
                    with local_activation_dfb.wait() as activation_block:
                        ttl.copy(
                            activation_block,
                            gathered_activation[
                                m_begin:m_end, gathered_k_begin:gathered_k_end
                            ],
                        ).wait()

                def receive_activation_from_device(pipe) -> None:
                    source_index = pipe.source_device_index
                    source_k_begin = source_index * k_tiles_per_device
                    k_begin = source_k_begin + transfer_index * k_tiles_per_transfer
                    k_end = k_begin + k_tiles_per_transfer
                    with fabric_receive_dfb.reserve() as activation_block:
                        ttl.copy(pipe, activation_block).wait()
                    with fabric_receive_dfb.wait() as activation_block:
                        ttl.copy(
                            activation_block,
                            gathered_activation[m_begin:m_end, k_begin:k_end],
                        ).wait()

                for transfer_index in range(k_transfer_count):
                    activation_all_gather_net.if_dst(receive_activation_from_device)

            for source_index in range(device_count):
                source_k_begin = source_index * k_tiles_per_device
                for transfer_index in range(k_transfer_count):
                    k_begin = source_k_begin + transfer_index * k_tiles_per_transfer
                    k_end = k_begin + k_tiles_per_transfer
                    with activation_compute_dfb.reserve() as activation_block:
                        if node_column == 0:
                            ttl.copy(
                                gathered_activation[m_begin:m_end, k_begin:k_end],
                                activation_block,
                            ).wait()

                            def send_activation_to_row(pipe) -> None:
                                ttl.copy(activation_block, pipe).wait()

                            activation_row_net.if_src(send_activation_to_row)
                        else:

                            def receive_activation_from_row(pipe) -> None:
                                ttl.copy(pipe, activation_block).wait()

                            activation_row_net.if_dst(receive_activation_from_row)

            n_begin = node_column * n_block_tiles
            n_end = n_begin + n_block_tiles
            with output_dfb.wait() as output_block:
                ttl.copy(
                    output_block,
                    output_shard[m_begin:m_end, n_begin:n_end],
                ).wait()

        @ttl.compute()
        def compute_output_block() -> None:
            with output_dfb.reserve() as output_block:
                accumulator = ttl.block.fill(
                    0.0, shape=output_block.shape, dtype=output_block.dtype
                )
                for _source_index in range(device_count):
                    for _transfer_index in range(k_transfer_count):
                        with (
                            activation_compute_dfb.wait() as activation_block,
                            weight_compute_dfb.wait() as weight_block,
                        ):
                            accumulator += activation_block @ weight_block

                with bias_dfb.wait() as bias_block:
                    accumulator += ttl.block.broadcast(
                        bias_block,
                        dims=[0],
                        shape=(m_block_tiles, n_block_tiles),
                    )
                output_block.store(accumulator)

    return all_gather_minimal_matmul
