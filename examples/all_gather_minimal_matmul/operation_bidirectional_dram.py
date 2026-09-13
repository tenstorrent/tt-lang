# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""N-sharded matmul with a bidirectional DRAM activation all-gather.

    column 0, rows 0..3: exchange complete activation blocks in both ring
                         directions, stage remote blocks in DRAM, and inject
                         every completed block into its compute chain
    columns 1..M-workers: forward activations down N-worker compute chains
                          stream weights -> bias-initialized matmul -> output DRAM

    alternate the two-hop direction by K block to balance both fabric links

    communication and compute execute concurrently through bounded L1 DFBs
    cache full K across N rounds when enabled; otherwise stream K blocks

Run from the repository root (four devices, 120 compute and two communication
workers/device):
    python -m examples.all_gather_minimal_matmul --mesh-shape 4x1 \
        --compute-grid 12 10 --communication-workers 2 \
        --m-tiles 296 --k-tiles-per-device 40 \
        --n-tiles 480 --m-block-tiles 5 --k-block-tiles 10 \
        --n-block-tiles 12 --no-reuse-activation

Kernels: all_gather_minimal_matmul_bidirectional_dram below the network and DFB
declarations.
"""

from collections.abc import Callable

import ttl
import ttnn

from examples.all_gather_minimal_matmul.config import AllGatherMinimalMatmulConfig
from examples.all_gather_minimal_matmul.collectives import make_ring_graph


def make_bidirectional_dram_all_gather_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
    communication_worker_count: int = 2,
) -> Callable[..., None]:
    if config.device_count != 4:
        raise ValueError(
            "bidirectional DRAM all-gather currently requires four devices"
        )
    if not 1 <= communication_worker_count <= min(config.n_workers, config.m_workers):
        raise ValueError("communication workers must fit the communication column")
    if config.m_workers % communication_worker_count != 0:
        raise ValueError("compute M workers must be divisible by communication workers")
    device_domain = ttl.DeviceDomain(config.mesh_shape)
    fabric_worker_count = communication_worker_count
    fabric_worker_nodes = tuple(
        (0, fabric_worker_index) for fabric_worker_index in range(fabric_worker_count)
    )
    activation_forward_net = ttl.PipeNet(
        graph=make_ring_graph(device_domain, config.mesh_shape),
        local_nodes=fabric_worker_nodes,
    )
    activation_backward_net = ttl.PipeNet(
        graph=make_ring_graph(device_domain, config.mesh_shape, reverse=True),
        local_nodes=fabric_worker_nodes,
    )
    n_worker_count, m_worker_count = config.n_workers, config.m_workers
    fabric_served_row_count = m_worker_count // fabric_worker_count
    activation_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, fabric_worker_index),
                dst=(
                    served_row * fabric_worker_count + fabric_worker_index + 1,
                    slice(0, n_worker_count),
                ),
            )
            for served_row in range(fabric_served_row_count)
            for fabric_worker_index in range(fabric_worker_count)
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
    compute_k_tiles = config.k_block_tiles
    compute_k_blocks_per_device = k_tiles_per_device // compute_k_tiles
    activation_transfer_indices = tuple(range(device_count))
    logical_m_tiles = config.m_tiles
    activation_storage_rows = config.padded_m_tiles * 32
    m_rounds = config.padded_m_tiles // (m_block_tiles * m_worker_count)
    n_rounds = config.n_tiles_per_device // (n_block_tiles * n_worker_count)
    activation_gather_rounds = 1
    activation_block_count = config.activation_block_count
    reuse_activation = config.reuse_activation

    @ttl.operation(
        grid=(m_worker_count + 1, n_worker_count),
        device_domain=device_domain,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    def all_gather_minimal_matmul_bidirectional_dram(
        activation_shard: ttnn.Tensor,
        gathered_activation: ttnn.Tensor,
        weight_shard: ttnn.Tensor,
        bias_shard: ttnn.Tensor,
        output_shard: ttnn.Tensor,
    ) -> None:
        if activation_shard.shape[0] < activation_storage_rows:
            raise ValueError("activation storage must include padded M rows")
        if gathered_activation.shape[0] < activation_storage_rows:
            raise ValueError("gathered activation storage must include padded M rows")
        full_k_elements = device_count * k_tiles_per_device * 32
        if gathered_activation.shape[1] < full_k_elements:
            raise ValueError(
                "gathered activation storage must include the complete K dimension"
            )
        activation_relay_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, compute_k_tiles),
            block_count=2,
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
            bias_shard, shape=(1, n_block_tiles), block_count=1
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output_shard, shape=(m_block_tiles, n_block_tiles), block_count=1
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
            local_device_index = device_domain.current_index()
            if physical_column == 0 and physical_row < fabric_worker_count:
                for m_round in range(m_rounds):
                    for _activation_round in range(activation_gather_rounds):
                        for k_block in range(compute_k_blocks_per_device):
                            local_k_begin = k_block * compute_k_tiles
                            local_k_end = local_k_begin + compute_k_tiles

                            def receive_activation_row(row_pipe):
                                target_m_worker = row_pipe.dst[0][0] - 1
                                staging_m_begin = (
                                    m_round * m_worker_count + target_m_worker
                                ) * m_block_tiles
                                staging_m_end = staging_m_begin + m_block_tiles
                                forward_source_device = (
                                    local_device_index + device_count - 1
                                ) % device_count
                                backward_source_device = (
                                    local_device_index + 1
                                ) % device_count
                                opposite_source_device = (
                                    local_device_index + 2
                                ) % device_count
                                forward_k_begin = (
                                    forward_source_device * k_tiles_per_device
                                    + local_k_begin
                                )
                                backward_k_begin = (
                                    backward_source_device * k_tiles_per_device
                                    + local_k_begin
                                )
                                opposite_k_begin = (
                                    opposite_source_device * k_tiles_per_device
                                    + local_k_begin
                                )
                                primary_forward_staging = gathered_activation[
                                    staging_m_begin:staging_m_end,
                                    forward_k_begin : forward_k_begin + compute_k_tiles,
                                ]
                                primary_backward_staging = gathered_activation[
                                    staging_m_begin:staging_m_end,
                                    backward_k_begin : backward_k_begin
                                    + compute_k_tiles,
                                ]
                                secondary_forward_staging = gathered_activation[
                                    staging_m_begin:staging_m_end,
                                    forward_k_begin : forward_k_begin + compute_k_tiles,
                                ]
                                secondary_backward_staging = gathered_activation[
                                    staging_m_begin:staging_m_end,
                                    backward_k_begin : backward_k_begin
                                    + compute_k_tiles,
                                ]
                                opposite_forward_staging = gathered_activation[
                                    staging_m_begin:staging_m_end,
                                    opposite_k_begin : opposite_k_begin
                                    + compute_k_tiles,
                                ]
                                opposite_backward_staging = gathered_activation[
                                    staging_m_begin:staging_m_end,
                                    opposite_k_begin : opposite_k_begin
                                    + compute_k_tiles,
                                ]
                                m_begin = staging_m_begin
                                local_relay = activation_relay_dfb.reserve()
                                ttl.copy(
                                    activation_shard[
                                        m_begin : m_begin + m_block_tiles,
                                        local_k_begin:local_k_end,
                                    ],
                                    local_relay,
                                ).wait()

                                def receive_primary_forward(pipe):
                                    receive_request = ttl.copy(
                                        pipe,
                                        primary_forward_staging,
                                        shape=(
                                            m_block_tiles,
                                            compute_k_tiles,
                                        ),
                                    )
                                    receive_request.wait()

                                def receive_primary_backward(pipe):
                                    receive_request = ttl.copy(
                                        pipe,
                                        primary_backward_staging,
                                        shape=(
                                            m_block_tiles,
                                            compute_k_tiles,
                                        ),
                                    )
                                    receive_request.wait()

                                if physical_row % 2 == 0:
                                    activation_forward_net.if_dst(
                                        receive_primary_forward
                                    )
                                    primary_relay = activation_relay_dfb.reserve()
                                    ttl.copy(
                                        primary_forward_staging,
                                        primary_relay,
                                    ).wait()
                                else:
                                    activation_backward_net.if_dst(
                                        receive_primary_backward
                                    )
                                    primary_relay = activation_relay_dfb.reserve()
                                    ttl.copy(
                                        primary_backward_staging,
                                        primary_relay,
                                    ).wait()

                                def receive_secondary_forward(pipe):
                                    receive_request = ttl.copy(
                                        pipe,
                                        secondary_forward_staging,
                                        shape=(
                                            m_block_tiles,
                                            compute_k_tiles,
                                        ),
                                    )
                                    receive_request.wait()

                                def receive_secondary_backward(pipe):
                                    receive_request = ttl.copy(
                                        pipe,
                                        secondary_backward_staging,
                                        shape=(
                                            m_block_tiles,
                                            compute_k_tiles,
                                        ),
                                    )
                                    receive_request.wait()

                                if physical_row % 2 == 0:
                                    activation_backward_net.if_dst(
                                        receive_secondary_backward
                                    )
                                    secondary_relay = activation_relay_dfb.reserve()
                                    ttl.copy(
                                        secondary_backward_staging,
                                        secondary_relay,
                                    ).wait()
                                else:
                                    activation_forward_net.if_dst(
                                        receive_secondary_forward
                                    )
                                    secondary_relay = activation_relay_dfb.reserve()
                                    ttl.copy(
                                        secondary_forward_staging,
                                        secondary_relay,
                                    ).wait()

                                def receive_opposite_forward(pipe):
                                    receive_request = ttl.copy(
                                        pipe,
                                        opposite_forward_staging,
                                        shape=(
                                            m_block_tiles,
                                            compute_k_tiles,
                                        ),
                                    )
                                    receive_request.wait()

                                def receive_opposite_backward(pipe):
                                    receive_request = ttl.copy(
                                        pipe,
                                        opposite_backward_staging,
                                        shape=(
                                            m_block_tiles,
                                            compute_k_tiles,
                                        ),
                                    )
                                    receive_request.wait()

                                if physical_row % 2 == 0:
                                    activation_forward_net.if_dst(
                                        receive_opposite_forward
                                    )
                                    opposite_relay = activation_relay_dfb.reserve()
                                    ttl.copy(
                                        opposite_forward_staging,
                                        opposite_relay,
                                    ).wait()
                                else:
                                    activation_backward_net.if_dst(
                                        receive_opposite_backward
                                    )
                                    opposite_relay = activation_relay_dfb.reserve()
                                    ttl.copy(
                                        opposite_backward_staging,
                                        opposite_relay,
                                    ).wait()

                            activation_row_net.if_src(receive_activation_row)
            if physical_column > 0:
                m_worker_index = physical_column - 1
                n_worker_index = physical_row
                for m_round in range(m_rounds):
                    m_begin = (
                        m_round * m_worker_count + m_worker_index
                    ) * m_block_tiles
                    for n_round in range(n_rounds):
                        if n_round == 0:
                            for _local_k_block in range(compute_k_blocks_per_device):
                                for _transfer_index in activation_transfer_indices:
                                    activation_block = matmul_activation_dfb.reserve()

                                    def receive_activation(pipe):
                                        ttl.copy(pipe, activation_block).wait()

                                    activation_row_net.if_dst(receive_activation)
                        else:
                            for local_k_block in range(compute_k_blocks_per_device):
                                local_k_begin = local_k_block * compute_k_tiles
                                for transfer_index in activation_transfer_indices:
                                    activation_block = matmul_activation_dfb.reserve()
                                    if not reuse_activation:
                                        source_device_index = local_device_index
                                        if transfer_index == 1:
                                            source_device_index = (
                                                local_device_index + device_count - 1
                                            ) % device_count
                                        elif transfer_index == 2:
                                            source_device_index = (
                                                local_device_index + 1
                                            ) % device_count
                                        elif transfer_index == 3:
                                            source_device_index = (
                                                local_device_index + 2
                                            ) % device_count
                                        if transfer_index == 0:
                                            ttl.copy(
                                                activation_shard[
                                                    m_begin : m_begin + m_block_tiles,
                                                    local_k_begin : local_k_begin
                                                    + compute_k_tiles,
                                                ],
                                                activation_block,
                                            ).wait()
                                        else:
                                            gathered_k_begin = (
                                                source_device_index * k_tiles_per_device
                                                + local_k_begin
                                            )
                                            ttl.copy(
                                                gathered_activation[
                                                    m_begin : m_begin + m_block_tiles,
                                                    gathered_k_begin : gathered_k_begin
                                                    + compute_k_tiles,
                                                ],
                                                activation_block,
                                            ).wait()
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
            if physical_column == 0 and physical_row < fabric_worker_count:
                for m_round in range(m_rounds):
                    for _activation_round in range(activation_gather_rounds):
                        for k_block in range(compute_k_blocks_per_device):

                            def send_activation_row(row_pipe):
                                local_relay = activation_relay_dfb.wait()
                                primary_send_reservation = (
                                    matmul_activation_dfb.reserve()
                                )
                                ttl.copy(
                                    local_relay,
                                    primary_send_reservation,
                                    byte_count=activation_block_bytes,
                                ).wait()
                                secondary_send_reservation = (
                                    matmul_activation_dfb.reserve()
                                )
                                ttl.copy(
                                    local_relay,
                                    secondary_send_reservation,
                                    byte_count=activation_block_bytes,
                                ).wait()
                                ttl.copy(local_relay, row_pipe).wait()
                                primary_send_block = matmul_activation_dfb.wait()

                                def send_primary(pipe):
                                    ttl.copy(primary_send_block, pipe).wait()

                                if physical_row % 2 == 0:
                                    activation_forward_net.if_src(send_primary)
                                else:
                                    activation_backward_net.if_src(send_primary)

                                primary_relay = activation_relay_dfb.wait()
                                primary_forward_reservation = (
                                    matmul_activation_dfb.reserve()
                                )
                                ttl.copy(
                                    primary_relay,
                                    primary_forward_reservation,
                                    byte_count=activation_block_bytes,
                                ).wait()
                                if physical_row % 2 == 0:
                                    ttl.copy(primary_relay, row_pipe).wait()
                                secondary_send_block = matmul_activation_dfb.wait()

                                def send_secondary(pipe):
                                    ttl.copy(secondary_send_block, pipe).wait()

                                if physical_row % 2 == 0:
                                    activation_backward_net.if_src(send_secondary)
                                else:
                                    activation_forward_net.if_src(send_secondary)

                                primary_forward_block = matmul_activation_dfb.wait()

                                def forward_primary(pipe):
                                    ttl.copy(primary_forward_block, pipe).wait()

                                if physical_row % 2 == 0:
                                    activation_forward_net.if_src(forward_primary)
                                else:
                                    activation_backward_net.if_src(forward_primary)

                                secondary_relay = activation_relay_dfb.wait()
                                ttl.copy(secondary_relay, row_pipe).wait()
                                if physical_row % 2 != 0:
                                    ttl.copy(primary_forward_block, row_pipe).wait()
                                opposite_relay = activation_relay_dfb.wait()
                                ttl.copy(opposite_relay, row_pipe).wait()

                            activation_row_net.if_src(send_activation_row)
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
                            for transfer_index in range(device_count):
                                source_device_index = local_device_index
                                if transfer_index == 1:
                                    source_device_index = (
                                        local_device_index + device_count - 1
                                    ) % device_count
                                elif transfer_index == 2:
                                    source_device_index = (
                                        local_device_index + 1
                                    ) % device_count
                                elif transfer_index == 3:
                                    source_device_index = (
                                        local_device_index + 2
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

    return all_gather_minimal_matmul_bidirectional_dram
