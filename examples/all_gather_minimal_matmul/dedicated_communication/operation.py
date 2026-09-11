# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""N-sharded matmul with dedicated activation communication workers.

    physical column 0: receive activation rows -> multicast to compute rows
                       forward rows to the next device through the existing ring
    columns 1..M-workers: stream weights -> bias-initialized matmul -> output DRAM

    communication and compute execute concurrently through bounded L1 DFBs
    cache full K across N rounds when enabled; otherwise stream K blocks

Run from the repository root (four devices, 120 compute and four communication
workers/device):
    python -m examples.all_gather_minimal_matmul.n_sharded --mesh-shape 2x2 \
        --worker-grid 12 10 --transpose --dedicated-communication-workers 4 \
        --activation-all-gather ring --m-tiles 24 --k-tiles-per-device 4 \
        --n-tiles-per-device 20 --m-block-tiles 2 --k-block-tiles 2 \
        --n-block-tiles 2 --no-reuse-activation

Kernels: all_gather_minimal_matmul below the network and DFB declarations.
"""


from collections.abc import Callable

import ttl
import ttnn

from examples.all_gather_minimal_matmul.config import AllGatherMinimalMatmulConfig
from examples.all_gather_minimal_matmul.collectives import make_all_gather_graph


def make_all_gather_minimal_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
    all_gather_algorithm: str = "ring",
    communication_worker_count: int = 4,
) -> Callable[..., None]:
    if (
        not config.transpose
        or config.device_count < 2
        or all_gather_algorithm != "ring"
    ):
        raise ValueError(
            "dedicated communication requires a transposed multi-device ring"
        )
    if not 1 <= communication_worker_count <= min(config.n_workers, config.m_workers):
        raise ValueError("communication workers must fit the communication column")
    maximum_communication_workers = 2 if config.device_count == 2 else 4
    if communication_worker_count > maximum_communication_workers:
        raise ValueError(
            f"{config.device_count}-device ring supports at most "
            f"{maximum_communication_workers} dedicated communication workers"
        )
    device_domain = ttl.DeviceDomain(config.mesh_shape)
    activation_all_gather_net = ttl.PipeNet(
        graph=make_all_gather_graph(device_domain, config.mesh_shape, "ring")
    )
    n_worker_count, m_worker_count = config.n_workers, config.m_workers
    served_row_count = (
        m_worker_count + communication_worker_count - 1
    ) // communication_worker_count
    activation_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, m_worker_index % communication_worker_count),
                dst=(m_worker_index + 1, slice(0, n_worker_count)),
            )
            for m_worker_index in range(m_worker_count)
        ]
    )
    weight_column_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(1, n_worker_index),
                dst=(slice(2, m_worker_count + 1), n_worker_index),
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
    activation_read_rounds = 1 if config.reuse_activation else n_rounds
    activation_block_count = config.activation_block_count

    @ttl.operation(
        grid=(m_worker_count + 1, n_worker_count),
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
            raise ValueError("activation storage must include padded M rows")
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
            accumulation_dtype, shape=(m_block_tiles, n_block_tiles), block_count=1
        )
        accumulation_bias_dfb = ttl.make_dfb(
            accumulation_dtype, shape=(1, n_block_tiles), block_count=1
        )
        activation_block_bytes = (
            m_block_tiles
            * compute_k_tiles
            * activation_shard.get_tile().get_tile_size(activation_shard.dtype)
        )

        @ttl.datamovement()
        def receive_activations_and_write_output():
            physical_column, physical_row = ttl.node(dims=2)
            if physical_column == 0 and physical_row < communication_worker_count:
                local_served_row_count = (
                    m_worker_count + communication_worker_count - 1 - physical_row
                ) // communication_worker_count
                for m_round in range(m_rounds):
                    for _activation_round in range(activation_read_rounds):
                        for k_block in range(compute_k_blocks_per_device):
                            local_k_begin = k_block * compute_k_tiles
                            local_k_end = local_k_begin + compute_k_tiles
                            for source_round in range(device_count):
                                for served_row in range(local_served_row_count):
                                    target_m_worker = (
                                        served_row * communication_worker_count
                                        + physical_row
                                    )
                                    m_begin = (
                                        m_round * m_worker_count + target_m_worker
                                    ) * m_block_tiles
                                    row_block = activation_row_staging_dfb.reserve()
                                    if source_round == 0:
                                        ttl.copy(
                                            activation_shard[
                                                m_begin : m_begin + m_block_tiles,
                                                local_k_begin:local_k_end,
                                            ],
                                            row_block,
                                        ).wait()
                                    else:

                                        def receive_from_previous_device(pipe):
                                            received = (
                                                remote_activation_receive_dfb.reserve()
                                            )
                                            ttl.copy(pipe, received).wait()

                                        activation_all_gather_net.if_dst(
                                            receive_from_previous_device
                                        )
                                        received = remote_activation_receive_dfb.wait()
                                        ttl.copy(
                                            received,
                                            row_block,
                                            byte_count=activation_block_bytes,
                                        ).wait()

                                def multicast_to_compute_row(pipe):
                                    row_block = activation_row_staging_dfb.wait()
                                    if source_round < device_count - 1:
                                        relay = activation_relay_dfb.reserve()
                                        ttl.copy(
                                            row_block,
                                            relay,
                                            byte_count=activation_block_bytes,
                                        ).wait()
                                    ttl.copy(row_block, pipe).wait()

                                activation_row_net.if_src(multicast_to_compute_row)
            if physical_column > 0:
                m_worker_index = physical_column - 1
                n_worker_index = physical_row
                for m_round in range(m_rounds):
                    m_begin = (
                        m_round * m_worker_count + m_worker_index
                    ) * m_block_tiles
                    for n_round in range(n_rounds):
                        for k_block in range(
                            device_count * compute_k_blocks_per_device
                        ):
                            activation_block = matmul_activation_dfb.reserve()
                            if n_round < activation_read_rounds:

                                def receive_from_communication_worker(pipe):
                                    received = (
                                        broadcast_activation_receive_dfb.reserve()
                                    )
                                    ttl.copy(pipe, received).wait()
                                    received = broadcast_activation_receive_dfb.wait()
                                    ttl.copy(
                                        received,
                                        activation_block,
                                        byte_count=activation_block_bytes,
                                    ).wait()

                                activation_row_net.if_dst(
                                    receive_from_communication_worker
                                )
                        n_begin = (
                            n_round * n_worker_count + n_worker_index
                        ) * n_block_tiles
                        output_block = output_dfb.wait()
                        if m_begin < logical_m_tiles:
                            ttl.copy(
                                output_block,
                                output_shard[
                                    m_begin : m_begin + m_block_tiles,
                                    n_begin : n_begin + n_block_tiles,
                                ],
                            ).wait()

        @ttl.datamovement()
        def forward_activations_and_distribute_weights():
            physical_column, physical_row = ttl.node(dims=2)
            local_device_index = device_domain.current_index()
            if physical_column == 0 and physical_row < communication_worker_count:
                local_served_row_count = (
                    m_worker_count + communication_worker_count - 1 - physical_row
                ) // communication_worker_count
                for m_round in range(m_rounds):
                    for _activation_round in range(activation_read_rounds):
                        for k_block in range(compute_k_blocks_per_device):
                            for source_round in range(device_count - 1):
                                for served_row in range(local_served_row_count):
                                    relay = activation_relay_dfb.wait()

                                    def forward_to_next_device(pipe):
                                        ttl.copy(relay, pipe).wait()

                                    activation_all_gather_net.if_src(
                                        forward_to_next_device
                                    )
            if physical_column > 0:
                m_worker_index = physical_column - 1
                n_worker_index = physical_row
                for m_round in range(m_rounds):
                    for n_round in range(n_rounds):
                        n_begin = (
                            n_round * n_worker_count + n_worker_index
                        ) * n_block_tiles
                        bias_block = bias_dfb.reserve()
                        ttl.copy(
                            bias_shard[0:1, n_begin : n_begin + n_block_tiles],
                            bias_block,
                        ).wait()
                        for k_block in range(compute_k_blocks_per_device):
                            for source_round in range(device_count):
                                source_device_index = (
                                    local_device_index + device_count - source_round
                                ) % device_count
                                k_begin = (
                                    source_device_index * k_tiles_per_device
                                    + k_block * compute_k_tiles
                                )
                                weight_block = matmul_weight_dfb.reserve()
                                if m_worker_index == 0:
                                    ttl.copy(
                                        weight_shard[
                                            k_begin : k_begin + compute_k_tiles,
                                            n_begin : n_begin + n_block_tiles,
                                        ],
                                        weight_block,
                                    ).wait()

                                    def multicast_weight(pipe):
                                        ttl.copy(weight_block, pipe).wait()

                                    weight_column_net.if_src(multicast_weight)
                                else:

                                    def receive_weight(pipe):
                                        ttl.copy(pipe, weight_block).wait()

                                    weight_column_net.if_dst(receive_weight)

        @ttl.compute()
        def compute_matmul_and_bias():
            physical_column, physical_row = ttl.node(dims=2)
            if physical_column > 0:
                for m_round in range(m_rounds):
                    for n_round in range(n_rounds):
                        bias_block = bias_dfb.wait()
                        converted_bias = accumulation_bias_dfb.reserve()
                        converted_bias.store(
                            ttl.math.typecast(bias_block, converted_bias.dtype)
                        )
                        converted_bias = accumulation_bias_dfb.wait()
                        accumulator = matmul_accumulator_dfb.reserve()
                        accumulator.store(
                            ttl.block.broadcast(
                                converted_bias,
                                dims=[0],
                                shape=(m_block_tiles, n_block_tiles),
                            )
                        )
                        for k_block in range(
                            device_count * compute_k_blocks_per_device
                        ):
                            activation_block = matmul_activation_dfb.wait()
                            weight_block = matmul_weight_dfb.wait()
                            accumulator += ttl.math.typecast(
                                activation_block @ weight_block, accumulator.dtype
                            )
                        accumulator = matmul_accumulator_dfb.wait()
                        output_block = output_dfb.reserve()
                        output_block.store(
                            ttl.math.typecast(accumulator, output_block.dtype)
                        )

    return all_gather_minimal_matmul
