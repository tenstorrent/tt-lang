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

Run from the repository root (four devices, 120 compute and four fabric
workers/device):
    python -m examples.all_gather_minimal_matmul --mesh-shape 4x1 \
        --compute-grid 12 10 --communication-workers 4 \
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
    communication_worker_count: int = 4,
) -> Callable[..., None]:
    if config.device_count != 4:
        raise ValueError(
            "bidirectional DRAM all-gather currently requires four devices"
        )
    if not 1 <= communication_worker_count <= min(config.n_workers, config.m_workers):
        raise ValueError("communication workers must fit the communication column")
    device_domain = ttl.DeviceDomain(config.mesh_shape)
    fabric_worker_count = communication_worker_count
    activation_forward_net = ttl.PipeNet(
        graph=make_ring_graph(device_domain, config.mesh_shape)
    )
    activation_backward_net = ttl.PipeNet(
        graph=make_ring_graph(device_domain, config.mesh_shape, reverse=True)
    )
    n_worker_count, m_worker_count = config.n_workers, config.m_workers
    fabric_served_row_count = (
        m_worker_count + fabric_worker_count - 1
    ) // fabric_worker_count
    direct_activation_entry_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, m_worker_index % fabric_worker_count),
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
        else direct_activation_entry_net
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
    def all_gather_minimal_matmul_bidirectional_dram(
        activation_shard: ttnn.Tensor,
        gathered_activation: ttnn.Tensor,
        weight_shard: ttnn.Tensor,
        bias_shard: ttnn.Tensor,
        output_shard: ttnn.Tensor,
    ) -> None:
        if activation_shard.shape[0] < activation_storage_rows:
            raise ValueError("activation storage must include padded M rows")
        staging_storage_rows = fabric_worker_count * m_block_tiles * 32
        if gathered_activation.shape[0] < staging_storage_rows:
            raise ValueError(
                "activation staging storage must include one row block per "
                "communication worker"
            )
        if gathered_activation.shape[1] < 4 * compute_k_tiles * 32:
            raise ValueError(
                "activation staging storage must include four transfer slots"
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

        @ttl.datamovement()
        def receive_activations_and_write_output():
            physical_column, physical_row = ttl.node(dims=2)
            if physical_column == 0 and physical_row < fabric_worker_count:
                staging_m_begin = physical_row * m_block_tiles
                staging_m_end = staging_m_begin + m_block_tiles
                forward_staging = gathered_activation[
                    staging_m_begin:staging_m_end, 0:compute_k_tiles
                ]
                backward_staging = gathered_activation[
                    staging_m_begin:staging_m_end,
                    compute_k_tiles : 2 * compute_k_tiles,
                ]
                opposite_forward_staging = gathered_activation[
                    staging_m_begin:staging_m_end,
                    2 * compute_k_tiles : 3 * compute_k_tiles,
                ]
                opposite_backward_staging = gathered_activation[
                    staging_m_begin:staging_m_end,
                    3 * compute_k_tiles : 4 * compute_k_tiles,
                ]
                local_served_row_count = (
                    m_worker_count + fabric_worker_count - 1 - physical_row
                ) // fabric_worker_count
                for m_round in range(m_rounds):
                    for _activation_round in range(activation_read_rounds):
                        for k_block in range(compute_k_blocks_per_device):
                            local_k_begin = k_block * compute_k_tiles
                            local_k_end = local_k_begin + compute_k_tiles
                            for served_row in range(local_served_row_count):
                                target_m_worker = (
                                    served_row * fabric_worker_count + physical_row
                                )
                                m_begin = (
                                    m_round * m_worker_count + target_m_worker
                                ) * m_block_tiles
                                for transfer_index in range(device_count):
                                    if transfer_index == 0:
                                        relay = activation_relay_dfb.reserve()
                                        ttl.copy(
                                            activation_shard[
                                                m_begin : m_begin + m_block_tiles,
                                                local_k_begin:local_k_end,
                                            ],
                                            relay,
                                        ).wait()
                                    else:

                                        def receive_forward(pipe):
                                            receive_request = ttl.copy(
                                                pipe,
                                                forward_staging,
                                                shape=(
                                                    m_block_tiles,
                                                    compute_k_tiles,
                                                ),
                                            )
                                            receive_request.wait()

                                        def receive_backward(pipe):
                                            receive_request = ttl.copy(
                                                pipe,
                                                backward_staging,
                                                shape=(
                                                    m_block_tiles,
                                                    compute_k_tiles,
                                                ),
                                            )
                                            receive_request.wait()

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

                                        if transfer_index == 1:
                                            activation_forward_net.if_dst(
                                                receive_forward
                                            )
                                        elif transfer_index == 2:
                                            activation_backward_net.if_dst(
                                                receive_backward
                                            )
                                        elif k_block % 2 == 0:
                                            activation_forward_net.if_dst(
                                                receive_opposite_forward
                                            )
                                        else:
                                            activation_backward_net.if_dst(
                                                receive_opposite_backward
                                            )
                                        relay = activation_relay_dfb.reserve()
                                        if transfer_index == 1:
                                            ttl.copy(forward_staging, relay).wait()
                                        elif transfer_index == 2:
                                            ttl.copy(backward_staging, relay).wait()
                                        elif k_block % 2 == 0:
                                            ttl.copy(
                                                opposite_forward_staging, relay
                                            ).wait()
                                        else:
                                            ttl.copy(
                                                opposite_backward_staging, relay
                                            ).wait()
            if physical_column > 0:
                m_worker_index = physical_column - 1
                n_worker_index = physical_row
                for m_round in range(m_rounds):
                    m_begin = (
                        m_round * m_worker_count + m_worker_index
                    ) * m_block_tiles
                    for n_round in range(n_rounds):
                        for _local_k_block in range(compute_k_blocks_per_device):
                            for _transfer_index in range(device_count):
                                activation_block = matmul_activation_dfb.reserve()
                                if n_round < activation_read_rounds:
                                    received_activation = activation_block

                                    def receive_activation(pipe):
                                        ttl.copy(pipe, received_activation).wait()

                                    if n_worker_index == 0:
                                        direct_activation_entry_net.if_dst(
                                            receive_activation
                                        )
                                    else:
                                        activation_compute_chain_net.if_dst(
                                            receive_activation
                                        )

                                    if n_worker_index < n_worker_count - 1:

                                        def forward_activation(pipe):
                                            ttl.copy(received_activation, pipe).wait()

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
            if physical_column == 0 and physical_row < fabric_worker_count:
                local_served_row_count = (
                    m_worker_count + fabric_worker_count - 1 - physical_row
                ) // fabric_worker_count
                for m_round in range(m_rounds):
                    for _activation_round in range(activation_read_rounds):
                        for k_block in range(compute_k_blocks_per_device):
                            for served_row in range(local_served_row_count):
                                for transfer_index in range(device_count):
                                    relay = activation_relay_dfb.wait()

                                    def send_activation(pipe):
                                        ttl.copy(relay, pipe).wait()

                                    direct_activation_entry_net.if_src(send_activation)
                                    if transfer_index == 0:
                                        activation_forward_net.if_src(send_activation)
                                        activation_backward_net.if_src(send_activation)
                                    elif transfer_index == 1 and k_block % 2 == 0:
                                        activation_forward_net.if_src(send_activation)
                                    elif transfer_index == 2 and k_block % 2 != 0:
                                        activation_backward_net.if_src(send_activation)
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
                                else:
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
