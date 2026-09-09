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
    worker_grid: tuple[int, int] | None = None
    transpose: bool = False

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

        if self.worker_grid is not None:
            object.__setattr__(self, "worker_grid", tuple(self.worker_grid))
            if len(self.worker_grid) != 2 or min(self.worker_grid) < 2:
                raise ValueError("worker_grid must have two extents of at least two")
        if (self.m_tiles // self.m_block_tiles) % self.m_workers:
            raise ValueError("M block count must be divisible by the M worker count")
        if (self.n_tiles_per_device // self.n_block_tiles) % self.n_workers:
            raise ValueError("N block count must be divisible by the N worker count")

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
        if self.worker_grid is not None:
            return self.worker_grid
        logical_grid = (
            self.n_tiles_per_device // self.n_block_tiles,
            self.m_tiles // self.m_block_tiles,
        )
        return logical_grid[::-1] if self.transpose else logical_grid

    @property
    def m_workers(self) -> int:
        return self.grid[0 if self.transpose else 1]

    @property
    def n_workers(self) -> int:
        return self.grid[1 if self.transpose else 0]

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

    grid_columns, grid_rows = config.n_workers, config.m_workers

    def worker_coordinates(column, row):
        return (row, column) if config.transpose else (column, row)

    activation_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=worker_coordinates(0, row_index),
                dst=worker_coordinates(slice(1, grid_columns), row_index),
            )
            for row_index in range(grid_rows)
        ]
    )
    weight_column_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=worker_coordinates(column_index, 0),
                dst=worker_coordinates(column_index, slice(1, grid_rows)),
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
    m_rounds = config.m_tiles // (m_block_tiles * grid_rows)
    n_rounds = config.n_tiles_per_device // (n_block_tiles * grid_columns)
    column_axis = 1 if config.transpose else 0
    row_axis = 0 if config.transpose else 1

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

        accumulation_dtype = ttnn.float32 if fp32_dest_acc_en else output_shard.dtype
        accumulator_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(m_block_tiles, n_block_tiles), block_count=2
        )
        partial_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(m_block_tiles, n_block_tiles), block_count=2
        )
        converted_bias_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(1, n_block_tiles), block_count=2
        )
        product_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(m_block_tiles, n_block_tiles), block_count=2
        )

        @ttl.datamovement()
        def send_activation_and_distribute_weights() -> None:
            physical_column, physical_row = ttl.node(dims=2)
            node_column = (
                physical_column * (1 - column_axis) + physical_row * column_axis
            )
            node_row = physical_column * (1 - row_axis) + physical_row * row_axis
            for m_round in range(m_rounds):
                m_begin = (m_round * grid_rows + node_row) * m_block_tiles
                m_end = m_begin + m_block_tiles

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

                for n_round in range(n_rounds):
                    n_begin = (n_round * grid_columns + node_column) * n_block_tiles
                    n_end = n_begin + n_block_tiles
                    for source_index in range(device_count):
                        source_k_begin = source_index * k_tiles_per_device
                        for transfer_index in range(k_transfer_count):
                            k_begin = (
                                source_k_begin + transfer_index * k_tiles_per_transfer
                            )
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
            physical_column, physical_row = ttl.node(dims=2)
            node_column = (
                physical_column * (1 - column_axis) + physical_row * column_axis
            )
            node_row = physical_column * (1 - row_axis) + physical_row * row_axis
            for m_round in range(m_rounds):
                m_begin = (m_round * grid_rows + node_row) * m_block_tiles
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
                                activation_shard[
                                    m_begin:m_end, local_k_begin:local_k_end
                                ],
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

                for n_round in range(n_rounds):
                    for source_index in range(device_count):
                        source_k_begin = source_index * k_tiles_per_device
                        for transfer_index in range(k_transfer_count):
                            k_begin = (
                                source_k_begin + transfer_index * k_tiles_per_transfer
                            )
                            k_end = k_begin + k_tiles_per_transfer
                            with activation_compute_dfb.reserve() as activation_block:
                                if node_column == 0:
                                    ttl.copy(
                                        gathered_activation[
                                            m_begin:m_end, k_begin:k_end
                                        ],
                                        activation_block,
                                    ).wait()

                                    def send_activation_to_row(pipe) -> None:
                                        ttl.copy(activation_block, pipe).wait()

                                    activation_row_net.if_src(send_activation_to_row)
                                else:

                                    def receive_activation_from_row(pipe) -> None:
                                        ttl.copy(pipe, activation_block).wait()

                                    activation_row_net.if_dst(
                                        receive_activation_from_row
                                    )

                    n_begin = (n_round * grid_columns + node_column) * n_block_tiles
                    n_end = n_begin + n_block_tiles
                    with output_dfb.wait() as output_block:
                        ttl.copy(
                            output_block,
                            output_shard[m_begin:m_end, n_begin:n_end],
                        ).wait()

        @ttl.compute()
        def compute_output_block() -> None:
            for _m_round in range(m_rounds):
                for _n_round in range(n_rounds):
                    with accumulator_dfb.reserve() as accumulator_block:
                        accumulator_block.store(
                            ttl.block.fill(
                                0.0,
                                shape=accumulator_block.shape,
                                dtype=accumulator_block.dtype,
                            )
                        )
                    for _k_transfer in range(device_count * k_transfer_count):
                        with (
                            activation_compute_dfb.wait() as activation_block,
                            weight_compute_dfb.wait() as weight_block,
                            product_dfb.reserve() as product_block,
                        ):
                            product_block.store(activation_block @ weight_block)
                        with (
                            product_dfb.wait() as product_block,
                            partial_dfb.reserve() as partial_block,
                        ):
                            partial_block.store(
                                ttl.math.typecast(product_block, partial_block.dtype)
                            )
                        with (
                            partial_dfb.wait() as partial_block,
                            accumulator_dfb.wait() as previous_accumulator,
                            accumulator_dfb.reserve() as next_accumulator,
                        ):
                            next_accumulator.store(previous_accumulator + partial_block)

                    with (
                        bias_dfb.wait() as bias_block,
                        converted_bias_dfb.reserve() as converted_bias,
                    ):
                        converted_bias.store(
                            ttl.math.typecast(bias_block, converted_bias.dtype)
                        )
                    with (
                        converted_bias_dfb.wait() as converted_bias,
                        accumulator_dfb.wait() as accumulator_block,
                        partial_dfb.reserve() as result_block,
                    ):
                        result_block.store(
                            accumulator_block
                            + ttl.block.broadcast(
                                converted_bias,
                                dims=[0],
                                shape=(m_block_tiles, n_block_tiles),
                            )
                        )
                    with (
                        partial_dfb.wait() as result_block,
                        output_dfb.reserve() as output_block,
                    ):
                        output_block.store(
                            ttl.math.typecast(result_block, output_block.dtype)
                        )

    return all_gather_minimal_matmul
