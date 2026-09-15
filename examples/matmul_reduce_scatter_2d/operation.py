# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Two-dimensional matmul with block-streamed reduce-scatter.

device mesh: P_K=2 by P_N
activation per device: M x K/P_K, replicated across P_N
weight per device: K/P_K x N/P_N
bias per device: 1 x N/P_N, FP32
matmul partial per device: M x N/P_N
output per device: M/P_K x N/P_N

data-movement thread 0: exchange partial blocks; write preceding reduced output
data-movement thread 1: distribute activation, weight, and bias inputs
compute: calculate local and peer blocks; reduce the preceding pair; add bias
"""

from collections.abc import Callable

import ttl
import ttnn

from .config import MatmulReduceScatter2DConfig


def make_matmul_reduce_scatter_2d_operation(
    config: MatmulReduceScatter2DConfig,
    *,
    math_fidelity: str | None = None,
    fp32_dest_acc_en: bool | None = None,
) -> Callable[..., None]:
    device_domain = ttl.DeviceDomain(config.mesh_shape)
    m_worker_count = config.m_worker_count
    n_worker_count = config.n_worker_count
    k_group_count = config.k_group_count
    n_group_count = config.n_group_count
    m_block_tiles = config.m_block_tiles
    k_block_tiles = config.k_block_tiles
    n_block_tiles = config.n_block_tiles
    k_block_count = config.k_tiles_per_group // k_block_tiles
    n_round_count = config.n_round_count
    output_block_count = config.output_block_count
    output_m_tiles_per_group = config.output_m_tiles_per_group

    reduce_to_group_zero_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain,
            edges=[
                ((1, n_group_index), (0, n_group_index))
                for n_group_index in range(n_group_count)
            ],
        )
    )
    reduce_to_group_one_net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(
            device_domain,
            edges=[
                ((0, n_group_index), (1, n_group_index))
                for n_group_index in range(n_group_count)
            ],
        )
    )
    activation_row_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(m_worker_index, 0),
                dst=(m_worker_index, slice(1, n_worker_count)),
            )
            for m_worker_index in range(m_worker_count)
        ]
    )
    weight_column_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, n_worker_index),
                dst=(slice(1, m_worker_count), n_worker_index),
            )
            for n_worker_index in range(n_worker_count)
        ]
    )

    @ttl.operation(
        grid=(m_worker_count, n_worker_count),
        device_domain=device_domain,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    def matmul_reduce_scatter_2d(
        activation_shard: ttnn.Tensor,
        weight_shard: ttnn.Tensor,
        bias_shard: ttnn.Tensor,
        output_shard: ttnn.Tensor,
    ) -> None:
        matmul_activation_dfb = ttl.make_dataflow_buffer_like(
            activation_shard,
            shape=(m_block_tiles, k_block_tiles),
            block_count=2,
        )
        matmul_weight_dfb = ttl.make_dataflow_buffer_like(
            weight_shard,
            shape=(k_block_tiles, n_block_tiles),
            block_count=2,
        )
        bias_dfb = ttl.make_dataflow_buffer_like(
            bias_shard, shape=(1, n_block_tiles), block_count=1
        )
        outgoing_partial_dfb = ttl.make_dataflow_buffer_like(
            output_shard,
            shape=(m_block_tiles, n_block_tiles),
            block_count=1,
        )
        remote_partial_dfb = ttl.make_dataflow_buffer_like(
            output_shard,
            shape=(m_block_tiles, n_block_tiles),
            block_count=1,
        )
        output_dfb = ttl.make_dataflow_buffer_like(
            output_shard,
            shape=(m_block_tiles, n_block_tiles),
            block_count=1,
        )
        accumulation_dtype = ttnn.float32 if fp32_dest_acc_en else output_shard.dtype
        scratch_accumulator_dfb = ttl.make_dfb(
            accumulation_dtype,
            shape=(m_block_tiles, n_block_tiles),
            block_count=1,
        )
        local_accumulator_dfb = ttl.make_dfb(
            accumulation_dtype,
            shape=(m_block_tiles, n_block_tiles),
            block_count=1,
        )
        partial_block_bytes = (
            m_block_tiles
            * n_block_tiles
            * output_shard.get_tile().get_tile_size(output_shard.dtype)
        )

        @ttl.datamovement()
        def exchange_partial_blocks():
            m_worker_index, n_worker_index = ttl.node(dims=2)
            for output_index in range(output_block_count):

                def receive_group_zero_partial(pipe):
                    remote_partial = remote_partial_dfb.reserve()
                    ttl.copy(
                        pipe,
                        remote_partial,
                        byte_count=partial_block_bytes,
                    ).wait()

                reduce_to_group_zero_net.if_dst(receive_group_zero_partial)

                def send_group_zero_partial(pipe):
                    outgoing_partial = outgoing_partial_dfb.wait()
                    ttl.copy(
                        outgoing_partial,
                        pipe,
                        byte_count=partial_block_bytes,
                    ).wait()

                reduce_to_group_zero_net.if_src(send_group_zero_partial)

                def receive_group_one_partial(pipe):
                    remote_partial = remote_partial_dfb.reserve()
                    ttl.copy(
                        pipe,
                        remote_partial,
                        byte_count=partial_block_bytes,
                    ).wait()

                reduce_to_group_one_net.if_dst(receive_group_one_partial)

                def send_group_one_partial(pipe):
                    outgoing_partial = outgoing_partial_dfb.wait()
                    ttl.copy(
                        outgoing_partial,
                        pipe,
                        byte_count=partial_block_bytes,
                    ).wait()

                reduce_to_group_one_net.if_src(send_group_one_partial)

                if output_index > 0:
                    previous_output_index = output_index - 1
                    previous_n_round = previous_output_index % n_round_count
                    previous_n_begin = (
                        previous_n_round * n_worker_count + n_worker_index
                    ) * n_block_tiles
                    previous_m_round = previous_output_index // n_round_count
                    previous_local_m_begin = (
                        previous_m_round * m_worker_count + m_worker_index
                    ) * m_block_tiles
                    output_block = output_dfb.wait()
                    ttl.copy(
                        output_block,
                        output_shard[
                            previous_local_m_begin : previous_local_m_begin
                            + m_block_tiles,
                            previous_n_begin : previous_n_begin + n_block_tiles,
                        ],
                    ).wait()

            final_output_index = output_block_count - 1
            final_m_round = final_output_index // n_round_count
            final_n_round = final_output_index % n_round_count
            final_local_m_begin = (
                final_m_round * m_worker_count + m_worker_index
            ) * m_block_tiles
            final_n_begin = (
                final_n_round * n_worker_count + n_worker_index
            ) * n_block_tiles
            output_block = output_dfb.wait()
            ttl.copy(
                output_block,
                output_shard[
                    final_local_m_begin : final_local_m_begin + m_block_tiles,
                    final_n_begin : final_n_begin + n_block_tiles,
                ],
            ).wait()

        @ttl.datamovement()
        def distribute_inputs():
            m_worker_index, n_worker_index = ttl.node(dims=2)
            local_device_index = device_domain.current_index()
            local_k_group_index = local_device_index // n_group_count

            for output_index in range(output_block_count):
                m_round = output_index // n_round_count
                n_round = output_index % n_round_count
                n_begin = (n_round * n_worker_count + n_worker_index) * n_block_tiles
                local_m_begin = (
                    m_round * m_worker_count + m_worker_index
                ) * m_block_tiles
                for partial_index in range(k_group_count):
                    target_k_group_index = (
                        local_k_group_index + k_group_count - 1 - partial_index
                    ) % k_group_count
                    m_begin = (
                        target_k_group_index * output_m_tiles_per_group + local_m_begin
                    )
                    for k_block_index in range(k_block_count):
                        k_begin = k_block_index * k_block_tiles
                        activation_block = matmul_activation_dfb.reserve()
                        if n_worker_index == 0:
                            ttl.copy(
                                activation_shard[
                                    m_begin : m_begin + m_block_tiles,
                                    k_begin : k_begin + k_block_tiles,
                                ],
                                activation_block,
                            ).wait()

                            def multicast_activation(pipe):
                                ttl.copy(activation_block, pipe).wait()

                            activation_row_net.if_src(multicast_activation)
                        else:

                            def receive_activation(pipe):
                                ttl.copy(pipe, activation_block).wait()

                            activation_row_net.if_dst(receive_activation)

                        weight_block = matmul_weight_dfb.reserve()
                        if m_worker_index == 0:
                            ttl.copy(
                                weight_shard[
                                    k_begin : k_begin + k_block_tiles,
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

                bias_block = bias_dfb.reserve()
                ttl.copy(
                    bias_shard[0:1, n_begin : n_begin + n_block_tiles],
                    bias_block,
                ).wait()

        @ttl.compute()
        def compute_partial_and_reduce():
            for _output_index in range(output_block_count):
                outgoing_accumulator = scratch_accumulator_dfb.reserve()
                outgoing_accumulator.store(
                    ttl.block.fill(
                        0.0,
                        shape=outgoing_accumulator.shape,
                        dtype=outgoing_accumulator.dtype,
                    )
                )
                for _k_block_index in range(k_block_count):
                    activation_block = matmul_activation_dfb.wait()
                    weight_block = matmul_weight_dfb.wait()
                    outgoing_accumulator += ttl.math.matmul(
                        activation_block,
                        weight_block,
                        dtype=outgoing_accumulator.dtype,
                    )
                outgoing_accumulator = scratch_accumulator_dfb.wait()
                outgoing_partial = outgoing_partial_dfb.reserve()
                outgoing_partial.store(
                    ttl.math.typecast(outgoing_accumulator, outgoing_partial.dtype)
                )

                if _output_index > 0:
                    previous_local_partial = local_accumulator_dfb.wait()
                    previous_remote_partial = remote_partial_dfb.wait()
                    reduction_accumulator = scratch_accumulator_dfb.reserve()
                    reduction_accumulator.store(
                        ttl.math.typecast(
                            previous_remote_partial,
                            reduction_accumulator.dtype,
                        )
                    )
                    previous_bias_block = bias_dfb.wait()
                    for _reduce_partial in range(1):
                        reduction_accumulator += (
                            previous_local_partial
                            + ttl.block.broadcast(
                                previous_bias_block,
                                dims=[0],
                                shape=(m_block_tiles, n_block_tiles),
                            )
                        )
                    reduction_accumulator = scratch_accumulator_dfb.wait()
                    previous_output_block = output_dfb.reserve()
                    previous_output_block.store(
                        ttl.math.typecast(
                            reduction_accumulator,
                            previous_output_block.dtype,
                        )
                    )

                local_accumulator = local_accumulator_dfb.reserve()
                local_accumulator.store(
                    ttl.block.fill(
                        0.0,
                        shape=local_accumulator.shape,
                        dtype=local_accumulator.dtype,
                    )
                )
                for _k_block_index in range(k_block_count):
                    activation_block = matmul_activation_dfb.wait()
                    weight_block = matmul_weight_dfb.wait()
                    local_accumulator += ttl.math.matmul(
                        activation_block,
                        weight_block,
                        dtype=local_accumulator.dtype,
                    )

            final_local_partial = local_accumulator_dfb.wait()
            final_remote_partial = remote_partial_dfb.wait()
            reduction_accumulator = scratch_accumulator_dfb.reserve()
            reduction_accumulator.store(
                ttl.math.typecast(
                    final_remote_partial,
                    reduction_accumulator.dtype,
                )
            )
            final_bias_block = bias_dfb.wait()
            for _reduce_partial in range(1):
                reduction_accumulator += final_local_partial + ttl.block.broadcast(
                    final_bias_block,
                    dims=[0],
                    shape=(m_block_tiles, n_block_tiles),
                )
            reduction_accumulator = scratch_accumulator_dfb.wait()
            final_output_block = output_dfb.reserve()
            final_output_block.store(
                ttl.math.typecast(
                    reduction_accumulator,
                    final_output_block.dtype,
                )
            )

    return matmul_reduce_scatter_2d
