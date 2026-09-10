# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compute ``all_gather(activation) @ weight + bias`` across a fabric mesh.

Activation is sharded along K; weight, row bias, and output are sharded along N.
Each device gathers activation blocks into L1, broadcasts them across its N
workers, and multiplies them by weight blocks broadcast across its M workers.
Cached activations are reused across N blocks. Compute and data movement run
in separate concurrent kernels; bias is added after the full K reduction.
Inputs and outputs use BF16 or FP32 tensors in TILE layout and DRAM storage.

The TT-Lang implementation is the nested ``all_gather_minimal_matmul`` function
inside ``make_all_gather_minimal_matmul_operation``, below the configuration
class and worker-network setup. Its three kernels receive activations/write
outputs, send activations/distribute weights, and compute matmul plus bias.

Run from the repository root in an activated, fabric-enabled TT-Lang container,
with all visible devices idle (the default discovers the participant mesh)::

    timeout 300 python -m examples.all_gather_minimal_matmul 2>&1 | tee /tmp/device_test.log

See ``README.md`` beside this file for dimensions and mesh selection.
"""

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
    n_block_tiles: int = 1
    worker_grid: tuple[int, int] | None = None
    transpose: bool = False
    k_block_tiles: int = 1
    reuse_activation: bool = True

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
            "k_block_tiles": self.k_block_tiles,
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
                "k_block_tiles",
                self.k_block_tiles,
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

        if self.reuse_activation and self.activation_block_count > 32:
            raise ValueError(
                "activation reuse requires at most 32 K blocks; increase "
                "k_block_tiles or disable reuse_activation"
            )

        if self.worker_grid is not None:
            object.__setattr__(self, "worker_grid", tuple(self.worker_grid))
            if len(self.worker_grid) != 2 or min(self.worker_grid) < 2:
                raise ValueError("worker_grid must have two extents of at least two")
        if (self.m_tiles // self.m_block_tiles) % self.m_workers:
            raise ValueError("M block count must be divisible by the M worker count")
        if (self.n_tiles_per_device // self.n_block_tiles) % self.n_workers:
            raise ValueError("N block count must be divisible by the N worker count")

        if min(self.grid) < 2:
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
    def compute_k_tiles(self) -> int:
        return self.k_block_tiles

    @property
    def activation_block_count(self) -> int:
        if not self.reuse_activation:
            return 2
        return self.device_count * self.k_tiles_per_device // self.k_block_tiles


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

    n_worker_count, m_worker_count = config.n_workers, config.m_workers

    def physical_worker_coordinates(n_worker, m_worker):
        return (m_worker, n_worker) if config.transpose else (n_worker, m_worker)

    activation_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=physical_worker_coordinates(0, m_worker_index),
                dst=physical_worker_coordinates(
                    slice(1, n_worker_count), m_worker_index
                ),
            )
            for m_worker_index in range(m_worker_count)
        ]
    )
    weight_column_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=physical_worker_coordinates(n_worker_index, 0),
                dst=physical_worker_coordinates(
                    n_worker_index, slice(1, m_worker_count)
                ),
            )
            for n_worker_index in range(n_worker_count)
        ]
    )

    m_block_tiles = config.m_block_tiles
    k_tiles_per_device = config.k_tiles_per_device
    n_block_tiles = config.n_block_tiles
    device_count = config.device_count
    compute_k_tiles = config.compute_k_tiles
    compute_k_blocks_per_device = k_tiles_per_device // compute_k_tiles
    m_rounds = config.m_tiles // (m_block_tiles * m_worker_count)
    n_rounds = config.n_tiles_per_device // (n_block_tiles * n_worker_count)
    reuse_activation = config.reuse_activation
    activation_block_count = config.activation_block_count
    activation_read_rounds = 1 if reuse_activation else n_rounds
    n_worker_axis = 1 if config.transpose else 0
    m_worker_axis = 0 if config.transpose else 1

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
        output_shard: ttnn.Tensor,
    ) -> None:
        local_activation_send_dfb = ttl.make_dataflow_buffer_like(
            activation_shard, shape=(m_block_tiles, compute_k_tiles), block_count=1
        )
        remote_activation_receive_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, compute_k_tiles),
            block_count=device_count - 1,
        )
        broadcast_activation_receive_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, compute_k_tiles),
            block_count=1,
        )
        matmul_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, compute_k_tiles),
            block_count=activation_block_count,
        )
        matmul_weight_dfb = ttl.make_dataflow_buffer_like(
            weight_shard,
            shape=(compute_k_tiles, n_block_tiles),
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
        matmul_accumulator_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(m_block_tiles, n_block_tiles), block_count=2
        )
        biased_accumulator_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(m_block_tiles, n_block_tiles), block_count=2
        )
        accumulation_bias_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(1, n_block_tiles), block_count=2
        )
        activation_block_bytes = (
            m_block_tiles
            * compute_k_tiles
            * activation_shard.get_tile().get_tile_size(activation_shard.dtype)
        )

        # Declaration order keeps activation traffic on the reader's NoC 0.
        @ttl.datamovement()
        def receive_activation_and_write_output() -> None:
            physical_column, physical_row = ttl.node(dims=2)
            n_worker_index = (
                physical_column * (1 - n_worker_axis) + physical_row * n_worker_axis
            )
            m_worker_index = (
                physical_column * (1 - m_worker_axis) + physical_row * m_worker_axis
            )
            local_device_index = device_domain.current_index()
            for m_round in range(m_rounds):
                m_begin = (m_round * m_worker_count + m_worker_index) * m_block_tiles
                m_end = m_begin + m_block_tiles
                for n_round in range(n_rounds):
                    for k_block in range(compute_k_blocks_per_device):
                        local_k_begin = k_block * compute_k_tiles
                        local_k_end = local_k_begin + compute_k_tiles
                        if n_round < activation_read_rounds and n_worker_index == 0:

                            def receive_activation_from_device(pipe) -> None:
                                received_block = remote_activation_receive_dfb.reserve()
                                ttl.copy(pipe, received_block).wait()

                            activation_all_gather_net.if_dst(
                                receive_activation_from_device
                            )
                        for source_device_index in range(device_count):
                            if n_round < activation_read_rounds:
                                activation_block = matmul_activation_dfb.reserve()
                                if n_worker_index == 0:
                                    if source_device_index == local_device_index:
                                        ttl.copy(
                                            activation_shard[
                                                m_begin:m_end,
                                                local_k_begin:local_k_end,
                                            ],
                                            activation_block,
                                        ).wait()
                                    else:
                                        received_block = (
                                            remote_activation_receive_dfb.wait()
                                        )
                                        ttl.copy(
                                            received_block,
                                            activation_block,
                                            byte_count=activation_block_bytes,
                                        ).wait()

                                    def send_activation_to_row(pipe) -> None:
                                        ttl.copy(activation_block, pipe).wait()

                                    activation_row_net.if_src(send_activation_to_row)
                                else:

                                    def receive_activation_from_row(pipe) -> None:
                                        received_block = (
                                            broadcast_activation_receive_dfb.reserve()
                                        )
                                        ttl.copy(pipe, received_block).wait()
                                        received_block = (
                                            broadcast_activation_receive_dfb.wait()
                                        )
                                        ttl.copy(
                                            received_block,
                                            activation_block,
                                            byte_count=activation_block_bytes,
                                        ).wait()

                                    activation_row_net.if_dst(
                                        receive_activation_from_row
                                    )

                            else:
                                # Full-K capacity returns each producer reservation to its cached pages.
                                cached_block = matmul_activation_dfb.reserve()

                    n_begin = (
                        n_round * n_worker_count + n_worker_index
                    ) * n_block_tiles
                    n_end = n_begin + n_block_tiles
                    output_block = output_dfb.wait()
                    ttl.copy(
                        output_block, output_shard[m_begin:m_end, n_begin:n_end]
                    ).wait()

        @ttl.datamovement()
        def send_activation_and_distribute_weights() -> None:
            physical_column, physical_row = ttl.node(dims=2)
            n_worker_index = (
                physical_column * (1 - n_worker_axis) + physical_row * n_worker_axis
            )
            m_worker_index = (
                physical_column * (1 - m_worker_axis) + physical_row * m_worker_axis
            )
            for m_round in range(m_rounds):
                m_begin = (m_round * m_worker_count + m_worker_index) * m_block_tiles
                m_end = m_begin + m_block_tiles
                for n_round in range(n_rounds):
                    n_begin = (
                        n_round * n_worker_count + n_worker_index
                    ) * n_block_tiles
                    n_end = n_begin + n_block_tiles
                    for k_block in range(compute_k_blocks_per_device):
                        if n_round < activation_read_rounds and n_worker_index == 0:
                            activation_block = local_activation_send_dfb.reserve()
                            local_k_begin = k_block * compute_k_tiles
                            local_k_end = local_k_begin + compute_k_tiles
                            ttl.copy(
                                activation_shard[
                                    m_begin:m_end, local_k_begin:local_k_end
                                ],
                                activation_block,
                            ).wait()
                            activation_block = local_activation_send_dfb.wait()

                            def send_activation_to_device(pipe) -> None:
                                ttl.copy(activation_block, pipe).wait()

                            activation_all_gather_net.if_src(send_activation_to_device)

                        for source_device_index in range(device_count):
                            k_begin = (
                                source_device_index * k_tiles_per_device
                                + k_block * compute_k_tiles
                            )
                            k_end = k_begin + compute_k_tiles
                            weight_block = matmul_weight_dfb.reserve()
                            if m_worker_index == 0:
                                ttl.copy(
                                    weight_shard[k_begin:k_end, n_begin:n_end],
                                    weight_block,
                                ).wait()

                                def broadcast_weight_to_column(pipe) -> None:
                                    ttl.copy(weight_block, pipe).wait()

                                weight_column_net.if_src(broadcast_weight_to_column)
                            else:

                                def receive_weight_from_column(pipe) -> None:
                                    ttl.copy(pipe, weight_block).wait()

                                weight_column_net.if_dst(receive_weight_from_column)
                    bias_block = bias_dfb.reserve()
                    ttl.copy(bias_shard[0:1, n_begin:n_end], bias_block).wait()

        @ttl.compute()
        def compute_matmul_and_bias() -> None:
            for _m_round in range(m_rounds):
                for _n_round in range(n_rounds):
                    accumulator_block = matmul_accumulator_dfb.reserve()
                    accumulator_block.store(
                        ttl.block.fill(
                            0.0,
                            shape=accumulator_block.shape,
                            dtype=accumulator_block.dtype,
                        )
                    )
                    for _k_block in range(device_count * compute_k_blocks_per_device):
                        activation_block = matmul_activation_dfb.wait()
                        weight_block = matmul_weight_dfb.wait()
                        accumulator_block += ttl.math.typecast(
                            activation_block @ weight_block,
                            accumulator_block.dtype,
                        )

                    bias_block = bias_dfb.wait()
                    accumulation_bias_block = accumulation_bias_dfb.reserve()
                    accumulation_bias_block.store(
                        ttl.math.typecast(bias_block, accumulation_bias_block.dtype)
                    )
                    accumulation_bias_block = accumulation_bias_dfb.wait()
                    accumulator_block = matmul_accumulator_dfb.wait()
                    biased_accumulator_block = biased_accumulator_dfb.reserve()
                    biased_accumulator_block.store(
                        accumulator_block
                        + ttl.block.broadcast(
                            accumulation_bias_block,
                            dims=[0],
                            shape=(m_block_tiles, n_block_tiles),
                        )
                    )
                    biased_accumulator_block = biased_accumulator_dfb.wait()
                    output_block = output_dfb.reserve()
                    output_block.store(
                        ttl.math.typecast(biased_accumulator_block, output_block.dtype)
                    )

    return all_gather_minimal_matmul
