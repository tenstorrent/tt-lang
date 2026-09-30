# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""N-sharded matmul with grouped-row activation transfers.

    communication node c owns contiguous M-worker rows
    read one dense M-row group from the local activation shard
    send one group per K block and source device around the fabric ring
    extract each M block in L1 -> inject into its compute-row chain
    stream weights -> bias-initialized matmul -> N-sharded output DRAM

    fabric communication and matmul execute concurrently through bounded L1 DFBs

Run from the repository root (four devices):
    python -m benchmarks.all_gather_minimal_matmul \
        --implementation ttlang --ttlang-activation-strategy grouped-row-l1 \
        --mesh-shape 4x1 --ttlang-compute-grid 12 10 \
        --ttlang-communication-workers 4 --m-tiles 296 \
        --k-tiles-per-device 40 --n-tiles 480 \
        --ttlang-m-block-tiles 5 --ttlang-k-block-tiles 10 \
        --ttlang-n-block-tiles 12 --no-ttlang-reuse-activation

Kernels: all_gather_minimal_matmul_grouped_rows below the network declarations.
"""

from collections.abc import Callable

import ttl
import ttnn

from examples.all_gather_minimal_matmul.collectives import make_ring_graph
from examples.all_gather_minimal_matmul.config import AllGatherMinimalMatmulConfig


def make_grouped_row_all_gather_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
    communication_worker_count: int = 4,
) -> Callable[..., None]:
    direct_fabric_worker_limit = 2 if config.device_count == 2 else 4
    if not 1 <= communication_worker_count <= direct_fabric_worker_limit:
        raise ValueError(
            f"grouped-row transfers support at most {direct_fabric_worker_limit} "
            "communication workers for this device mesh"
        )
    if communication_worker_count > min(config.n_workers, config.m_workers):
        raise ValueError("communication workers must fit the communication column")
    if config.m_workers % communication_worker_count:
        raise ValueError("compute M workers must be divisible by communication workers")

    device_domain = ttl.DeviceDomain(config.mesh_shape)
    n_worker_count = config.n_workers
    m_worker_count = config.m_workers
    compute_rows_per_communication_worker = m_worker_count // communication_worker_count
    communication_worker_nodes = tuple(
        (0, communication_worker_index)
        for communication_worker_index in range(communication_worker_count)
    )
    activation_all_gather_net = ttl.PipeNet(
        [ttl.Pipe(src=node, dst=node) for node in communication_worker_nodes],
        graph=make_ring_graph(device_domain, config.mesh_shape),
    )
    activation_entry_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(
                    0,
                    m_worker_index // compute_rows_per_communication_worker,
                ),
                dst=(m_worker_index + 1, 0),
            )
            for m_worker_index in range(m_worker_count)
        ]
    )
    activation_compute_chain_net = (
        ttl.PipeNet(
            [
                ttl.Pipe(
                    src=(m_worker_index + 1, n_worker_index),
                    dst=(m_worker_index + 1, n_worker_index + 1),
                )
                for m_worker_index in range(m_worker_count)
                for n_worker_index in range(n_worker_count - 1)
            ]
        )
        if n_worker_count > 1
        else activation_entry_net
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
    grouped_m_tiles = compute_rows_per_communication_worker * m_block_tiles
    k_tiles_per_device = config.k_tiles_per_device
    n_block_tiles = config.n_block_tiles
    device_count = config.device_count
    compute_k_tiles = config.k_block_tiles
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
    def all_gather_minimal_matmul_grouped_rows(
        activation_shard: ttnn.Tensor,
        weight_shard: ttnn.Tensor,
        bias_shard: ttnn.Tensor,
        output_shard: ttnn.Tensor,
    ) -> None:
        if activation_shard.shape[0] < activation_storage_rows:
            raise ValueError("activation storage must include padded M rows")

        activation_group_relay_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(grouped_m_tiles, compute_k_tiles),
            block_count=1,
        )
        activation_group_staging_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(grouped_m_tiles, compute_k_tiles),
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
            bias_shard,
            shape=(1, n_block_tiles),
            block_count=1,
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output_shard,
            shape=(m_block_tiles, n_block_tiles),
            block_count=1,
        )
        accumulation_dtype = ttnn.float32 if fp32_dest_acc_en else output_shard.dtype
        matmul_accumulator_dfb = ttl.make_dfb(
            accumulation_dtype,
            shape=(m_block_tiles, n_block_tiles),
            block_count=1,
        )
        accumulation_bias_dfb = ttl.make_dfb(
            accumulation_dtype,
            shape=(1, n_block_tiles),
            block_count=1,
        )
        activation_group_bytes = (
            grouped_m_tiles
            * compute_k_tiles
            * activation_shard.get_tile().get_tile_size(activation_shard.dtype)
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
                first_m_worker = physical_row * compute_rows_per_communication_worker
                for m_round in range(m_rounds):
                    grouped_m_begin = (
                        m_round * m_worker_count + first_m_worker
                    ) * m_block_tiles
                    for _activation_round in range(activation_read_rounds):
                        for k_block in range(compute_k_blocks_per_device):
                            local_k_begin = k_block * compute_k_tiles
                            local_k_end = local_k_begin + compute_k_tiles
                            for source_round in range(device_count):
                                activation_group = (
                                    activation_group_staging_dfb.reserve()
                                )
                                if source_round == 0:
                                    ttl.copy(
                                        activation_shard[
                                            grouped_m_begin : grouped_m_begin
                                            + grouped_m_tiles,
                                            local_k_begin:local_k_end,
                                        ],
                                        activation_group,
                                    ).wait()
                                else:

                                    def receive_from_previous_device(pipe):
                                        ttl.copy(
                                            pipe,
                                            activation_group,
                                            byte_count=activation_group_bytes,
                                        ).wait()

                                    activation_all_gather_net.if_dst(
                                        receive_from_previous_device
                                    )

                                activation_group = activation_group_staging_dfb.wait()

                                if source_round < device_count - 1:
                                    relay_group = activation_group_relay_dfb.reserve()
                                    ttl.copy(
                                        activation_group,
                                        relay_group,
                                        byte_count=activation_group_bytes,
                                    ).wait()

                                def inject_activation_block(entry_pipe):
                                    target_m_worker = entry_pipe.dst[0] - 1
                                    group_row = target_m_worker - first_m_worker
                                    activation_block = ttl.block.subview(
                                        activation_group,
                                        offsets=(group_row * m_block_tiles, 0),
                                        shape=(m_block_tiles, compute_k_tiles),
                                    )
                                    ttl.copy(
                                        activation_block,
                                        entry_pipe,
                                        byte_count=activation_block_bytes,
                                    ).wait()

                                activation_entry_net.if_src(inject_activation_block)

            if physical_column > 0:
                m_worker_index = physical_column - 1
                n_worker_index = physical_row
                for m_round in range(m_rounds):
                    m_begin = (
                        m_round * m_worker_count + m_worker_index
                    ) * m_block_tiles
                    for n_round in range(n_rounds):
                        for _local_k_block in range(compute_k_blocks_per_device):
                            for _source_round in range(device_count):
                                activation_block = matmul_activation_dfb.reserve()
                                if n_round < activation_read_rounds:
                                    if n_worker_index == 0:

                                        def receive_entry_activation(pipe):
                                            ttl.copy(
                                                pipe,
                                                activation_block,
                                                byte_count=activation_block_bytes,
                                            ).wait()

                                        activation_entry_net.if_dst(
                                            receive_entry_activation
                                        )
                                    else:

                                        def receive_chain_activation(pipe):
                                            ttl.copy(pipe, activation_block).wait()

                                        activation_compute_chain_net.if_dst(
                                            receive_chain_activation
                                        )

                                    if n_worker_index < n_worker_count - 1:

                                        def forward_activation(pipe):
                                            ttl.copy(activation_block, pipe).wait()

                                        activation_compute_chain_net.if_src(
                                            forward_activation
                                        )

                        n_begin = (
                            n_round * n_worker_count + n_worker_index
                        ) * n_block_tiles
                        output_block = output_dfb.wait()
                        if m_begin + m_block_tiles <= logical_m_tiles:
                            ttl.copy(
                                output_block,
                                output_shard[
                                    m_begin : m_begin + m_block_tiles,
                                    n_begin : n_begin + n_block_tiles,
                                ],
                            ).wait()
                        else:
                            for output_row in range(m_block_tiles):
                                if m_begin + output_row < logical_m_tiles:
                                    output_row_block = ttl.block.subview(
                                        output_block,
                                        offsets=(output_row, 0),
                                        shape=(1, n_block_tiles),
                                    )
                                    ttl.copy(
                                        output_row_block,
                                        output_shard[
                                            m_begin
                                            + output_row : m_begin
                                            + output_row
                                            + 1,
                                            n_begin : n_begin + n_block_tiles,
                                        ],
                                    ).wait()

        @ttl.datamovement()
        def forward_activations_and_distribute_weights():
            physical_column, physical_row = ttl.node(dims=2)
            local_device_index = device_domain.current_index()
            if physical_column == 0 and physical_row < communication_worker_count:
                for _m_round in range(m_rounds):
                    for _activation_round in range(activation_read_rounds):
                        for _k_block in range(compute_k_blocks_per_device):
                            for _source_round in range(device_count - 1):
                                relay_group = activation_group_relay_dfb.wait()

                                def forward_to_next_device(pipe):
                                    ttl.copy(
                                        relay_group,
                                        pipe,
                                        byte_count=activation_group_bytes,
                                    ).wait()

                                activation_all_gather_net.if_src(forward_to_next_device)

            if physical_column > 0:
                m_worker_index = physical_column - 1
                n_worker_index = physical_row
                for _m_round in range(m_rounds):
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
            physical_column, _physical_row = ttl.node(dims=2)
            if physical_column > 0:
                for _m_round in range(m_rounds):
                    for _n_round in range(n_rounds):
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
                        for _k_block in range(
                            device_count * compute_k_blocks_per_device
                        ):
                            activation_block = matmul_activation_dfb.wait()
                            weight_block = matmul_weight_dfb.wait()
                            accumulator += ttl.math.matmul(
                                activation_block,
                                weight_block,
                                dtype=accumulator.dtype,
                            )
                        accumulator = matmul_accumulator_dfb.wait()
                        output_block = output_dfb.reserve()
                        output_block.store(
                            ttl.math.typecast(accumulator, output_block.dtype)
                        )

    return all_gather_minimal_matmul_grouped_rows
