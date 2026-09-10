# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Ring activation exchange shared by two communication workers.

    receive each assigned row -> multicast across its N workers
    forward received rows -> next device
    stream weights -> matmul -> bias -> output DRAM

Run from the repository root:
    python -m examples.all_gather_minimal_matmul --mesh-shape 2x2 --worker-grid 2 3 --m-tiles 3 --n-tiles-per-device 2 --activation-all-gather ring

Kernels: all_gather_minimal_matmul below.
"""

from collections.abc import Callable

import ttl
import ttnn

from ..config import AllGatherMinimalMatmulConfig
from ..collectives import make_all_gather_graph


def make_all_gather_minimal_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
    all_gather_algorithm: str = "ring",
) -> Callable[..., None]:
    """Gather activation rows through two ring workers, then multicast to compute rows."""
    if (
        all_gather_algorithm != "ring"
        or config.device_count < 2
        or config.m_workers <= 2
    ):
        raise ValueError(
            "two-worker ring requires ring, multiple devices and more than two M workers"
        )
    device_domain = ttl.DeviceDomain(config.mesh_shape)
    activation_all_gather_net = ttl.PipeNet(
        graph=make_all_gather_graph(
            device_domain, config.mesh_shape, all_gather_algorithm
        )
    )
    (n_worker_count, m_worker_count) = (config.n_workers, config.m_workers)
    communication_worker_count = 2
    served_row_count = (
        m_worker_count + communication_worker_count - 1
    ) // communication_worker_count

    def physical_worker_coordinates(n_worker, m_worker):
        return (m_worker, n_worker) if config.transpose else (n_worker, m_worker)

    activation_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=physical_worker_coordinates(
                    0, m_worker_index % communication_worker_count
                ),
                dst=physical_worker_coordinates(
                    slice(
                        1 if m_worker_index < communication_worker_count else 0,
                        n_worker_count,
                    ),
                    m_worker_index,
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
    logical_m_tiles = config.m_tiles
    activation_storage_rows = config.padded_m_tiles * 32
    m_rounds = config.padded_m_tiles // (m_block_tiles * m_worker_count)
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
        if activation_shard.shape[0] < activation_storage_rows:
            raise ValueError(
                f"Activation storage requires {activation_storage_rows} rows for this worker grid"
            )
        remote_activation_receive_dfb = ttl.make_dataflow_buffer_like(
            activation_shard, shape=(m_block_tiles, compute_k_tiles), block_count=2
        )
        activation_relay_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, compute_k_tiles),
            block_count=max(2, served_row_count),
        )
        activation_row_staging_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, compute_k_tiles),
            block_count=served_row_count,
        )
        broadcast_activation_receive_dfb = ttl.make_dataflow_buffer_like(
            activation_shard, shape=(m_block_tiles, compute_k_tiles), block_count=1
        )
        matmul_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, compute_k_tiles),
            block_count=activation_block_count,
        )
        matmul_weight_dfb = ttl.make_dataflow_buffer_like(
            weight_shard, shape=(compute_k_tiles, n_block_tiles), block_count=2
        )
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias_shard, shape=(1, n_block_tiles), block_count=2
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(m_block_tiles, n_block_tiles), block_count=2
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

        @ttl.datamovement()
        def receive_activation_and_write_output() -> None:
            (physical_column, physical_row) = ttl.node(dims=2)
            n_worker_index = (
                physical_column * (1 - n_worker_axis) + physical_row * n_worker_axis
            )
            m_worker_index = (
                physical_column * (1 - m_worker_axis) + physical_row * m_worker_axis
            )
            local_served_row_count = (
                m_worker_count + communication_worker_count - 1 - m_worker_index
            ) // communication_worker_count
            local_device_index = device_domain.current_index()
            for m_round in range(m_rounds):
                m_begin = (m_round * m_worker_count + m_worker_index) * m_block_tiles
                m_end = m_begin + m_block_tiles
                for n_round in range(n_rounds):
                    for k_block in range(compute_k_blocks_per_device):
                        local_k_begin = k_block * compute_k_tiles
                        local_k_end = local_k_begin + compute_k_tiles
                        for source_round in range(device_count):
                            source_device_index = (
                                local_device_index + device_count - source_round
                            ) % device_count
                            if n_round < activation_read_rounds:
                                activation_block = matmul_activation_dfb.reserve()
                                if (
                                    n_worker_index == 0
                                    and m_worker_index < communication_worker_count
                                ):
                                    for served_row in range(local_served_row_count):
                                        target_m_worker = (
                                            served_row * communication_worker_count
                                            + m_worker_index
                                        )
                                        target_m_begin = (
                                            m_round * m_worker_count + target_m_worker
                                        ) * m_block_tiles
                                        target_m_end = target_m_begin + m_block_tiles
                                        row_block = activation_row_staging_dfb.reserve()
                                        if source_device_index == local_device_index:
                                            if served_row == 0:
                                                ttl.copy(
                                                    activation_shard[
                                                        target_m_begin:target_m_end,
                                                        local_k_begin:local_k_end,
                                                    ],
                                                    activation_block,
                                                ).wait()
                                            ttl.copy(
                                                activation_shard[
                                                    target_m_begin:target_m_end,
                                                    local_k_begin:local_k_end,
                                                ],
                                                row_block,
                                            ).wait()
                                        else:

                                            def receive_activation_from_source_device(
                                                fabric_pipe,
                                            ):
                                                received_block = (
                                                    remote_activation_receive_dfb.reserve()
                                                )
                                                ttl.copy(
                                                    fabric_pipe, received_block
                                                ).wait()

                                            activation_all_gather_net.if_dst(
                                                receive_activation_from_source_device
                                            )
                                            received_block = (
                                                remote_activation_receive_dfb.wait()
                                            )
                                            if served_row == 0:
                                                ttl.copy(
                                                    received_block,
                                                    activation_block,
                                                    byte_count=activation_block_bytes,
                                                ).wait()
                                            ttl.copy(
                                                received_block,
                                                row_block,
                                                byte_count=activation_block_bytes,
                                            ).wait()

                                    def broadcast_activation_row(pipe):
                                        row_block = activation_row_staging_dfb.wait()
                                        if source_round < device_count - 1:
                                            relay_block = activation_relay_dfb.reserve()
                                            ttl.copy(
                                                row_block,
                                                relay_block,
                                                byte_count=activation_block_bytes,
                                            ).wait()
                                        ttl.copy(row_block, pipe).wait()

                                    activation_row_net.if_src(broadcast_activation_row)
                                else:
                                    received_block = (
                                        broadcast_activation_receive_dfb.reserve()
                                    )

                                    def receive_activation_from_row(pipe):
                                        ttl.copy(pipe, received_block).wait()

                                    activation_row_net.if_dst(
                                        receive_activation_from_row
                                    )
                                    received_block = (
                                        broadcast_activation_receive_dfb.wait()
                                    )
                                    ttl.copy(
                                        received_block,
                                        activation_block,
                                        byte_count=activation_block_bytes,
                                    ).wait()
                            else:
                                cached_block = matmul_activation_dfb.reserve()
                    n_begin = (
                        n_round * n_worker_count + n_worker_index
                    ) * n_block_tiles
                    n_end = n_begin + n_block_tiles
                    output_block = output_dfb.wait()
                    if m_begin < logical_m_tiles:
                        ttl.copy(
                            output_block, output_shard[m_begin:m_end, n_begin:n_end]
                        ).wait()

        @ttl.datamovement()
        def send_activation_and_distribute_weights() -> None:
            (physical_column, physical_row) = ttl.node(dims=2)
            local_device_index = device_domain.current_index()
            n_worker_index = (
                physical_column * (1 - n_worker_axis) + physical_row * n_worker_axis
            )
            m_worker_index = (
                physical_column * (1 - m_worker_axis) + physical_row * m_worker_axis
            )
            local_served_row_count = (
                m_worker_count + communication_worker_count - 1 - m_worker_index
            ) // communication_worker_count
            for m_round in range(m_rounds):
                m_begin = (m_round * m_worker_count + m_worker_index) * m_block_tiles
                m_end = m_begin + m_block_tiles
                for n_round in range(n_rounds):
                    n_begin = (
                        n_round * n_worker_count + n_worker_index
                    ) * n_block_tiles
                    n_end = n_begin + n_block_tiles
                    for k_block in range(compute_k_blocks_per_device):
                        for source_round in range(device_count):
                            source_device_index = (
                                local_device_index + device_count - source_round
                            ) % device_count
                            if (
                                n_round < activation_read_rounds
                                and n_worker_index == 0
                                and (m_worker_index < communication_worker_count)
                                and (source_round > 0)
                            ):
                                for _served_row in range(local_served_row_count):
                                    relay_block = activation_relay_dfb.wait()

                                    def forward_activation_row(pipe):
                                        ttl.copy(relay_block, pipe).wait()

                                    activation_all_gather_net.if_src(
                                        forward_activation_row
                                    )
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
                            activation_block @ weight_block, accumulator_block.dtype
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
