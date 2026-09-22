# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""N-sharded matmul with bidirectional DRAM activation transport.

    forward client row: receive and forward right K halves
    backward client row: receive and forward left K halves
    row 0: read completed DRAM blocks and distribute them to the compute rows
    all compute workers: stream weights -> bias-initialized matmul -> output DRAM

    communicate only in the first N round; later N rounds reuse staged DRAM

    data-movement thread 0: activation transport, distribution, output write
    data-movement thread 1: weights and bias

Run from the repository root (four devices):
    python -m benchmarks.all_gather_minimal_matmul \
        --implementation ttlang --ttlang-activation-strategy bidirectional-dram \
        --mesh-shape 4x1 --ttlang-compute-grid 12 10 \
        --m-tiles 296 --k-tiles-per-device 40 --n-tiles 480 \
        --ttlang-m-block-tiles 5 --ttlang-k-block-tiles 10 \
        --ttlang-n-block-tiles 12 --no-ttlang-reuse-activation

Kernel: all_gather_minimal_matmul_bidirectional_dram below.
"""

from collections.abc import Callable

import ttl
import ttnn

from examples.all_gather_minimal_matmul.collectives import make_ring_graph
from examples.all_gather_minimal_matmul.config import AllGatherMinimalMatmulConfig


def make_bidirectional_dram_all_gather_matmul_operation(
    config: AllGatherMinimalMatmulConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
) -> Callable[..., None]:
    if config.device_count != 4:
        raise ValueError(
            "bidirectional DRAM all-gather currently requires four devices"
        )
    if config.k_block_tiles % 2:
        raise ValueError("bidirectional DRAM all-gather requires an even K block")
    if config.n_workers < 3:
        raise ValueError(
            "bidirectional DRAM all-gather requires three distinct compute rows"
        )
    if config.reuse_activation:
        raise ValueError(
            "bidirectional DRAM all-gather currently requires streamed activations"
        )

    device_domain = ttl.DeviceDomain(config.mesh_shape)
    m_worker_count = config.m_workers
    n_worker_count = config.n_workers
    assembly_row = 0
    forward_client_row = n_worker_count - 2
    backward_client_row = n_worker_count - 1
    forward_client_nodes = tuple(
        (m_worker_index, forward_client_row) for m_worker_index in range(m_worker_count)
    )
    backward_client_nodes = tuple(
        (m_worker_index, backward_client_row)
        for m_worker_index in range(m_worker_count)
    )
    activation_forward_net = ttl.PipeNet(
        [ttl.Pipe(src=node, dst=node) for node in forward_client_nodes],
        graph=make_ring_graph(device_domain, config.mesh_shape),
    )
    activation_backward_net = ttl.PipeNet(
        [ttl.Pipe(src=node, dst=node) for node in backward_client_nodes],
        graph=make_ring_graph(device_domain, config.mesh_shape, reverse=True),
    )
    forward_activation_to_assembly_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(m_worker_index, forward_client_row),
                dst=(m_worker_index, assembly_row),
            )
            for m_worker_index in range(m_worker_count)
        ]
    )
    backward_activation_to_assembly_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(m_worker_index, backward_client_row),
                dst=(m_worker_index, assembly_row),
            )
            for m_worker_index in range(m_worker_count)
        ]
    )
    activation_compute_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(m_worker_index, assembly_row),
                dst=(m_worker_index, slice(1, n_worker_count)),
            )
            for m_worker_index in range(m_worker_count)
        ]
    )
    weight_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, n_worker_index),
                dst=(slice(1, m_worker_count), n_worker_index),
            )
            for n_worker_index in range(n_worker_count)
        ]
    )

    m_block_tiles = config.m_block_tiles
    k_tiles_per_device = config.k_tiles_per_device
    n_block_tiles = config.n_block_tiles
    device_count = config.device_count
    compute_k_tiles = config.k_block_tiles
    half_k_tiles = compute_k_tiles // 2
    compute_k_blocks_per_device = k_tiles_per_device // compute_k_tiles
    logical_m_tiles = config.m_tiles
    activation_storage_rows = config.padded_m_tiles * 32
    m_rounds = config.padded_m_tiles // (m_block_tiles * m_worker_count)
    n_rounds = config.n_tiles_per_device // (n_block_tiles * n_worker_count)

    @ttl.operation(
        grid=(m_worker_count, n_worker_count),
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
            raise ValueError("activation staging must include padded M rows")
        if gathered_activation.shape[1] < device_count * k_tiles_per_device * 32:
            raise ValueError("activation staging must include the full K dimension")

        received_activation_half_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, half_k_tiles),
            block_count=1,
        )
        fabric_client_input_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, half_k_tiles),
            block_count=1,
        )
        fabric_send_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, half_k_tiles),
            block_count=1,
        )
        left_distribution_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, half_k_tiles),
            block_count=1,
        )
        right_distribution_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, half_k_tiles),
            block_count=1,
        )
        matmul_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, half_k_tiles),
            block_count=2,
        )
        left_weight_distribution_dfb = ttl.make_dataflow_buffer_like(
            weight_shard,
            shape=(half_k_tiles, n_block_tiles),
            block_count=1,
        )
        right_weight_distribution_dfb = ttl.make_dataflow_buffer_like(
            weight_shard,
            shape=(half_k_tiles, n_block_tiles),
            block_count=1,
        )
        matmul_weight_dfb = ttl.make_dataflow_buffer_like(
            weight_shard,
            shape=(half_k_tiles, n_block_tiles),
            block_count=2,
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
        activation_half_bytes = (
            m_block_tiles
            * half_k_tiles
            * activation_shard.get_tile().get_tile_size(activation_shard.dtype)
        )
        weight_half_bytes = (
            half_k_tiles
            * n_block_tiles
            * weight_shard.get_tile().get_tile_size(weight_shard.dtype)
        )

        @ttl.datamovement()
        def move_activations_and_write_output():
            m_worker_index, n_worker_index = ttl.node(dims=2)

            for m_round in range(m_rounds):
                m_begin = (m_round * m_worker_count + m_worker_index) * m_block_tiles
                for n_round in range(n_rounds):
                    for k_block in range(compute_k_blocks_per_device):
                        local_k_begin = k_block * compute_k_tiles
                        for source_distance in range(device_count):
                            staging_k_begin = (
                                source_distance * compute_k_blocks_per_device + k_block
                            ) * compute_k_tiles

                            if n_round == 0 and source_distance > 0:
                                if n_worker_index == forward_client_row:
                                    forward_staging = gathered_activation[
                                        m_begin : m_begin + m_block_tiles,
                                        staging_k_begin
                                        + half_k_tiles : staging_k_begin
                                        + compute_k_tiles,
                                    ]

                                    def receive_forward_half(pipe):
                                        ttl.copy(
                                            pipe,
                                            forward_staging,
                                            shape=(m_block_tiles, half_k_tiles),
                                        ).wait()

                                    activation_forward_net.if_dst(receive_forward_half)
                                    received_forward_half = (
                                        received_activation_half_dfb.reserve()
                                    )
                                    ttl.copy(
                                        forward_staging, received_forward_half
                                    ).wait()
                                    received_forward_half = (
                                        received_activation_half_dfb.wait()
                                    )

                                    def forward_received_activation(pipe):
                                        ttl.copy(received_forward_half, pipe).wait()

                                    forward_activation_to_assembly_net.if_src(
                                        forward_received_activation
                                    )

                                if n_worker_index == backward_client_row:
                                    backward_staging = gathered_activation[
                                        m_begin : m_begin + m_block_tiles,
                                        staging_k_begin : staging_k_begin
                                        + half_k_tiles,
                                    ]

                                    def receive_backward_half(pipe):
                                        ttl.copy(
                                            pipe,
                                            backward_staging,
                                            shape=(m_block_tiles, half_k_tiles),
                                        ).wait()

                                    activation_backward_net.if_dst(
                                        receive_backward_half
                                    )
                                    received_backward_half = (
                                        received_activation_half_dfb.reserve()
                                    )
                                    ttl.copy(
                                        backward_staging, received_backward_half
                                    ).wait()
                                    received_backward_half = (
                                        received_activation_half_dfb.wait()
                                    )

                                    def forward_received_backward_activation(pipe):
                                        ttl.copy(received_backward_half, pipe).wait()

                                    backward_activation_to_assembly_net.if_src(
                                        forward_received_backward_activation
                                    )

                            if n_worker_index == assembly_row:
                                left_activation = left_distribution_dfb.reserve()
                                if source_distance == 0:
                                    ttl.copy(
                                        activation_shard[
                                            m_begin : m_begin + m_block_tiles,
                                            local_k_begin : local_k_begin
                                            + half_k_tiles,
                                        ],
                                        left_activation,
                                    ).wait()
                                elif n_round == 0:

                                    def receive_left_activation(pipe):
                                        ttl.copy(pipe, left_activation).wait()

                                    backward_activation_to_assembly_net.if_dst(
                                        receive_left_activation
                                    )
                                else:
                                    ttl.copy(
                                        gathered_activation[
                                            m_begin : m_begin + m_block_tiles,
                                            staging_k_begin : staging_k_begin
                                            + half_k_tiles,
                                        ],
                                        left_activation,
                                    ).wait()

                                left_activation = left_distribution_dfb.wait()
                                compute_left_activation = (
                                    matmul_activation_dfb.reserve()
                                )
                                ttl.copy(
                                    left_activation,
                                    compute_left_activation,
                                    byte_count=activation_half_bytes,
                                ).wait()

                                right_activation = right_distribution_dfb.reserve()
                                if source_distance == 0:
                                    ttl.copy(
                                        activation_shard[
                                            m_begin : m_begin + m_block_tiles,
                                            local_k_begin
                                            + half_k_tiles : local_k_begin
                                            + compute_k_tiles,
                                        ],
                                        right_activation,
                                    ).wait()
                                elif n_round == 0:

                                    def receive_right_activation(pipe):
                                        ttl.copy(pipe, right_activation).wait()

                                    forward_activation_to_assembly_net.if_dst(
                                        receive_right_activation
                                    )
                                else:
                                    ttl.copy(
                                        gathered_activation[
                                            m_begin : m_begin + m_block_tiles,
                                            staging_k_begin
                                            + half_k_tiles : staging_k_begin
                                            + compute_k_tiles,
                                        ],
                                        right_activation,
                                    ).wait()

                                right_activation = right_distribution_dfb.wait()
                                compute_right_activation = (
                                    matmul_activation_dfb.reserve()
                                )
                                ttl.copy(
                                    right_activation,
                                    compute_right_activation,
                                    byte_count=activation_half_bytes,
                                ).wait()

                                def multicast_left_activation(pipe):
                                    ttl.copy(left_activation, pipe).wait()

                                activation_compute_net.if_src(multicast_left_activation)

                                def multicast_right_activation(pipe):
                                    ttl.copy(right_activation, pipe).wait()

                                activation_compute_net.if_src(
                                    multicast_right_activation
                                )
                            else:
                                client_left_activation = (
                                    fabric_client_input_dfb.reserve()
                                )

                                def receive_client_left_activation(pipe):
                                    ttl.copy(pipe, client_left_activation).wait()

                                activation_compute_net.if_dst(
                                    receive_client_left_activation
                                )
                                client_left_activation = fabric_client_input_dfb.wait()
                                compute_left_activation = (
                                    matmul_activation_dfb.reserve()
                                )
                                ttl.copy(
                                    client_left_activation,
                                    compute_left_activation,
                                    byte_count=activation_half_bytes,
                                ).wait()
                                if (
                                    n_round == 0
                                    and source_distance < device_count - 1
                                    and n_worker_index == backward_client_row
                                ):
                                    pending_backward_send = fabric_send_dfb.reserve()
                                    ttl.copy(
                                        client_left_activation,
                                        pending_backward_send,
                                        byte_count=activation_half_bytes,
                                    ).wait()
                                client_right_activation = (
                                    fabric_client_input_dfb.reserve()
                                )

                                def receive_client_right_activation(pipe):
                                    ttl.copy(pipe, client_right_activation).wait()

                                activation_compute_net.if_dst(
                                    receive_client_right_activation
                                )
                                client_right_activation = fabric_client_input_dfb.wait()
                                compute_right_activation = (
                                    matmul_activation_dfb.reserve()
                                )
                                ttl.copy(
                                    client_right_activation,
                                    compute_right_activation,
                                    byte_count=activation_half_bytes,
                                ).wait()
                                if (
                                    n_round == 0
                                    and source_distance < device_count - 1
                                    and n_worker_index == forward_client_row
                                ):
                                    pending_forward_send = fabric_send_dfb.reserve()
                                    ttl.copy(
                                        client_right_activation,
                                        pending_forward_send,
                                        byte_count=activation_half_bytes,
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
                                        m_begin + output_row : m_begin + output_row + 1,
                                        n_begin : n_begin + n_block_tiles,
                                    ],
                                ).wait()

        @ttl.datamovement()
        def read_bias_and_distribute_weights():
            m_worker_index, n_worker_index = ttl.node(dims=2)
            local_device_index = device_domain.current_index()
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
                        local_k_begin = k_block * compute_k_tiles
                        for source_distance in range(device_count):
                            if n_round == 0 and source_distance > 0:
                                if n_worker_index == forward_client_row:
                                    forward_fabric_send = fabric_send_dfb.wait()

                                    def send_forward_half(pipe):
                                        ttl.copy(forward_fabric_send, pipe).wait()

                                    activation_forward_net.if_src(send_forward_half)

                                if n_worker_index == backward_client_row:
                                    backward_fabric_send = fabric_send_dfb.wait()

                                    def send_backward_half(pipe):
                                        ttl.copy(backward_fabric_send, pipe).wait()

                                    activation_backward_net.if_src(send_backward_half)

                            source_forward = (
                                local_device_index + device_count - source_distance
                            ) % device_count
                            source_backward = (
                                local_device_index + source_distance
                            ) % device_count
                            left_weight = left_weight_distribution_dfb.reserve()
                            if m_worker_index == 0:
                                left_weight_begin = (
                                    source_backward * k_tiles_per_device + local_k_begin
                                )
                                ttl.copy(
                                    weight_shard[
                                        left_weight_begin : left_weight_begin
                                        + half_k_tiles,
                                        n_begin : n_begin + n_block_tiles,
                                    ],
                                    left_weight,
                                ).wait()

                                def multicast_left_weight(pipe):
                                    ttl.copy(left_weight, pipe).wait()

                                weight_row_net.if_src(multicast_left_weight)
                            else:

                                def receive_left_weight(pipe):
                                    ttl.copy(pipe, left_weight).wait()

                                weight_row_net.if_dst(receive_left_weight)

                            left_weight = left_weight_distribution_dfb.wait()
                            compute_left_weight = matmul_weight_dfb.reserve()
                            ttl.copy(
                                left_weight,
                                compute_left_weight,
                                byte_count=weight_half_bytes,
                            ).wait()

                            right_weight = right_weight_distribution_dfb.reserve()
                            if m_worker_index == 0:
                                right_weight_begin = (
                                    source_forward * k_tiles_per_device
                                    + local_k_begin
                                    + half_k_tiles
                                )
                                ttl.copy(
                                    weight_shard[
                                        right_weight_begin : right_weight_begin
                                        + half_k_tiles,
                                        n_begin : n_begin + n_block_tiles,
                                    ],
                                    right_weight,
                                ).wait()

                                def multicast_right_weight(pipe):
                                    ttl.copy(right_weight, pipe).wait()

                                weight_row_net.if_src(multicast_right_weight)
                            else:

                                def receive_right_weight(pipe):
                                    ttl.copy(pipe, right_weight).wait()

                                weight_row_net.if_dst(receive_right_weight)

                            right_weight = right_weight_distribution_dfb.wait()
                            compute_right_weight = matmul_weight_dfb.reserve()
                            ttl.copy(
                                right_weight,
                                compute_right_weight,
                                byte_count=weight_half_bytes,
                            ).wait()

        @ttl.compute()
        def compute_matmul_and_bias():
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
                    for _k_half in range(
                        2 * device_count * compute_k_blocks_per_device
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
