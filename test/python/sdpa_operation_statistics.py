# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.compile.log 2>&1
# RUN: ttlang-op-stats %t.initial.mlir | FileCheck %s

"""Exercise static operation statistics on a fused SDPA-style kernel.

The kernel contains two matrix multiplications, block softmax, multicast data
movement, and multiple loops. The test checks both exact aggregate counts and
unknown counts that require consumer-specific launch-coordinate facts.
"""

import torch
import ttnn

import ttl


TILE_SIZE = ttnn.TILE_SIZE
M_SIZE = K_SIZE = SCORE_SIZE = VALUE_SIZE = 512
M_BLOCK = N_BLOCK = K_BLOCK = 4
M_PARTITIONS = N_PARTITIONS = 4
K_ITERATIONS = K_SIZE // (K_BLOCK * TILE_SIZE)
SCORE_ITERATIONS = SCORE_SIZE // (N_BLOCK * TILE_SIZE)


@ttl.operation(grid=(N_PARTITIONS, M_PARTITIONS), fp32_dest_acc_en=False)
def sdpa_fused(query, key, value, output):
    query_dfb = ttl.make_dataflow_buffer_like(
        query, shape=(M_BLOCK, K_BLOCK), block_count=2
    )
    key_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(K_BLOCK, N_BLOCK), block_count=2
    )
    query_stage_dfb = ttl.make_dataflow_buffer_like(
        query, shape=(M_BLOCK, K_BLOCK), block_count=2
    )
    key_stage_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(K_BLOCK, N_BLOCK), block_count=2
    )
    score_dfb = ttl.make_dataflow_buffer_like(
        query, shape=(M_BLOCK, N_BLOCK), block_count=2
    )
    probability_dfb = ttl.make_dataflow_buffer_like(
        query, shape=(M_BLOCK, N_BLOCK), block_count=2
    )
    probability_receive_dfb = ttl.make_dataflow_buffer_like(
        value, shape=(M_BLOCK, N_BLOCK), block_count=SCORE_ITERATIONS
    )
    value_dfb = ttl.make_dataflow_buffer_like(
        value, shape=(N_BLOCK, N_BLOCK), block_count=2
    )
    value_stage_dfb = ttl.make_dataflow_buffer_like(
        value, shape=(N_BLOCK, N_BLOCK), block_count=2
    )
    output_dfb = ttl.make_dataflow_buffer_like(
        output, shape=(M_BLOCK, N_BLOCK), block_count=2
    )

    query_multicast = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, row_partition),
                dst=(slice(0, N_PARTITIONS), row_partition),
            )
            for row_partition in range(M_PARTITIONS)
        ]
    )
    key_multicast = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(column_partition, 0),
                dst=(column_partition, slice(0, M_PARTITIONS)),
            )
            for column_partition in range(N_PARTITIONS)
        ]
    )
    probability_gather = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(source_column, row_partition),
                dst=(slice(0, N_PARTITIONS), row_partition),
            )
            for row_partition in range(M_PARTITIONS)
            for source_column in range(N_PARTITIONS)
        ]
    )
    value_multicast = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(column_partition, 0),
                dst=(column_partition, slice(0, M_PARTITIONS)),
            )
            for column_partition in range(N_PARTITIONS)
        ]
    )

    @ttl.compute()
    def compute():
        score_block = score_dfb.reserve()
        for iteration in range(K_ITERATIONS):
            query_block = query_dfb.wait()
            key_block = key_dfb.wait()
            score_block += query_block @ key_block

        score_block = score_dfb.wait()
        maximum = ttl.math.reduce_max(score_block, dims=[0, 1])
        shifted = ttl.sub(
            score_block,
            ttl.block.broadcast(maximum, dims=[0, 1], shape=score_block.shape),
        )
        exponentials = ttl.exp(shifted)
        exponential_sum = ttl.math.reduce_sum(exponentials, dims=[0, 1])
        reciprocal_sum = ttl.recip(
            ttl.block.broadcast(exponential_sum, dims=[0, 1], shape=exponentials.shape)
        )
        probability_dfb.reserve().store(ttl.mul(exponentials, reciprocal_sum))

        output_block = output_dfb.reserve()
        for iteration in range(SCORE_ITERATIONS):
            probability_block = probability_receive_dfb.wait()
            value_block = value_dfb.wait()
            output_block += probability_block @ value_block

    @ttl.datamovement()
    def dm_brisc():
        column_coordinate, row_coordinate = ttl.node(dims=2)
        query_row = row_coordinate * M_BLOCK
        output_column = column_coordinate * N_BLOCK

        for iteration in range(K_ITERATIONS):
            reduction_column = iteration * K_BLOCK

            def send_query(pipe):
                staged_write = query_stage_dfb.reserve()
                ttl.copy(
                    query[
                        query_row : query_row + M_BLOCK,
                        reduction_column : reduction_column + K_BLOCK,
                    ],
                    staged_write,
                )
                staged_read = query_stage_dfb.wait()
                ttl.copy(staged_read, pipe)

            query_multicast.if_src(send_query)

            def send_key(pipe):
                staged_write = key_stage_dfb.reserve()
                ttl.copy(
                    key[
                        reduction_column : reduction_column + K_BLOCK,
                        output_column : output_column + N_BLOCK,
                    ],
                    staged_write,
                )
                staged_read = key_stage_dfb.wait()
                ttl.copy(staged_read, pipe)

            key_multicast.if_src(send_key)

        local_probability = probability_dfb.wait()

        def send_probability(pipe):
            ttl.copy(local_probability, pipe)

        probability_gather.if_src(send_probability)

        for iteration in range(SCORE_ITERATIONS):
            reduction_row = iteration * N_BLOCK

            def send_value(pipe):
                staged_write = value_stage_dfb.reserve()
                ttl.copy(
                    value[
                        reduction_row : reduction_row + N_BLOCK,
                        output_column : output_column + N_BLOCK,
                    ],
                    staged_write,
                )
                staged_read = value_stage_dfb.wait()
                ttl.copy(staged_read, pipe)

            value_multicast.if_src(send_value)

    @ttl.datamovement()
    def dm_ncrisc():
        column_coordinate, row_coordinate = ttl.node(dims=2)
        output_row = row_coordinate * M_BLOCK
        output_column = column_coordinate * N_BLOCK

        for iteration in range(K_ITERATIONS):

            def receive_query(pipe):
                ttl.copy(pipe, query_dfb.reserve())

            query_multicast.if_dst(receive_query)

            def receive_key(pipe):
                ttl.copy(pipe, key_dfb.reserve())

            key_multicast.if_dst(receive_key)

        def receive_probability(pipe):
            ttl.copy(pipe, probability_receive_dfb.reserve())

        probability_gather.if_dst(receive_probability)

        for iteration in range(SCORE_ITERATIONS):

            def receive_value(pipe):
                ttl.copy(pipe, value_dfb.reserve())

            value_multicast.if_dst(receive_value)

        output_block = output_dfb.wait()
        ttl.copy(
            output_block,
            output[
                output_row : output_row + M_BLOCK,
                output_column : output_column + N_BLOCK,
            ],
        )


# The compute function has two static matrix multiplications, each repeated by
# a four-iteration loop.
# CHECK-LABEL: func @compute
# CHECK:       scf.for static_occurrences=2 dynamic_instances=2
# CHECK:       ttl.attach_cb static_occurrences=8 dynamic_instances=20
# CHECK:       ttl.matmul static_occurrences=2 dynamic_instances=8
# CHECK:       ttl.store static_occurrences=3 dynamic_instances=9

# Launch-coordinate predicates are deliberately unknown to this toy client.
# Their nested data-movement operations therefore have unknown aggregate
# dynamic counts, while surrounding static loops remain exact.
# CHECK-LABEL: func @dm_brisc
# CHECK:       scf.yield static_occurrences=2 dynamic_instances=8
# CHECK:       ttl.copy static_occurrences=40 dynamic_instances=unknown
# CHECK:       ttl.create_pipe static_occurrences=28 dynamic_instances=64
# CHECK-LABEL: func @dm_ncrisc
# CHECK:       scf.yield static_occurrences=2 dynamic_instances=8
# CHECK:       ttl.copy static_occurrences=29 dynamic_instances=unknown
# CHECK:       ttl.create_pipe static_occurrences=28 dynamic_instances=64


def to_dram(device, tensor):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    device = ttnn.open_device(device_id=0)
    try:
        query = to_dram(device, torch.zeros((M_SIZE, K_SIZE), dtype=torch.bfloat16))
        key = to_dram(device, torch.zeros((K_SIZE, SCORE_SIZE), dtype=torch.bfloat16))
        value = to_dram(
            device, torch.zeros((SCORE_SIZE, VALUE_SIZE), dtype=torch.bfloat16)
        )
        output = to_dram(
            device, torch.zeros((M_SIZE, VALUE_SIZE), dtype=torch.bfloat16)
        )
        sdpa_fused(query, key, value, output)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
