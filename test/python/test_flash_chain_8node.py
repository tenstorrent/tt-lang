# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Eight-node flash-attention chain composed from operation atoms.

Each node computes streaming-softmax state for its K/V shard. A binary tree
combines the per-node states before node 0 normalizes and writes the result.
Frontend composition inlines both atoms into one operation whose logical DFB
count exceeds the hardware limit; concurrent-kernel liveness maps the logical DFBs
to a legal set of physical indices.
"""

import math

import pytest
import torch
import ttnn
import ttl

from ttlang_test_utils import is_hardware_available, to_dram
from utils.correctness import assert_pcc


TILE = ttnn.TILE_SIZE
PNHt = 1
DHt = 18
vDHt = 8
Sk_chunk_t = 4
N_CHUNKS = 1
NNODES = 8
# The generated Wormhole program exceeds the 96 KiB kernel-config default.
KERNEL_CONFIG_BUFFER_RESERVE_BYTES = 128 * 1024

St_per_node = Sk_chunk_t * N_CHUNKS
HEAD_DIM = DHt * TILE
HEAD_DIM_V = vDHt * TILE
SEQ_PER_NODE = St_per_node * TILE
SEQ = SEQ_PER_NODE * NNODES
Q_ROWS = PNHt * TILE
SCALE = 1.0 / math.sqrt(HEAD_DIM)
PCC_THRESHOLD = 0.99


if N_CHUNKS != 1:
    raise ValueError("flash_chain_8node requires N_CHUNKS == 1")


@ttl.operation()
def flash_shard_kd(
    query,
    key,
    value,
    local_max_dfb: ttl.DFB,
    local_sum_dfb: ttl.DFB,
    local_output_dfb: ttl.DFB,
):
    """Compute streaming-softmax state for one node's K/V shard."""
    query_dfb = ttl.make_dataflow_buffer_like(key, shape=(PNHt, DHt), block_count=2)
    key_dfb = ttl.make_dataflow_buffer_like(key, shape=(Sk_chunk_t, DHt), block_count=2)
    value_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(Sk_chunk_t, vDHt), block_count=2
    )
    transposed_key_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(DHt, Sk_chunk_t), block_count=2
    )
    scores_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, Sk_chunk_t), block_count=2
    )
    scores_for_reduce_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, Sk_chunk_t), block_count=2
    )
    scores_for_exp_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, Sk_chunk_t), block_count=2
    )
    max_broadcast_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, Sk_chunk_t), block_count=2
    )
    exp_scores_for_reduce_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, Sk_chunk_t), block_count=2
    )
    exp_scores_for_matmul_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, Sk_chunk_t), block_count=2
    )
    chunk_max_dfb = ttl.make_dataflow_buffer_like(key, shape=(PNHt, 1), block_count=2)
    merged_max_for_alpha_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, 1), block_count=2
    )
    merged_max_for_broadcast_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, 1), block_count=2
    )
    merged_max_for_state_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, 1), block_count=2
    )
    chunk_sum_dfb = ttl.make_dataflow_buffer_like(key, shape=(PNHt, 1), block_count=2)
    alpha_for_sum_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, 1), block_count=2
    )
    alpha_for_broadcast_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, 1), block_count=2
    )
    running_max_dfb = ttl.make_dataflow_buffer_like(key, shape=(PNHt, 1), block_count=2)
    running_sum_dfb = ttl.make_dataflow_buffer_like(key, shape=(PNHt, 1), block_count=2)
    alpha_broadcast_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, vDHt), block_count=2
    )
    corrected_output_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, vDHt), block_count=2
    )
    partial_value_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, vDHt), block_count=2
    )
    running_output_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, vDHt), block_count=2
    )

    initial_max = running_max_dfb.reserve()
    initial_max.store(ttl.block.fill(-1e30, shape=(PNHt, 1)))
    initial_sum = running_sum_dfb.reserve()
    initial_sum.store(ttl.block.fill(0, shape=(PNHt, 1)))
    initial_output = running_output_dfb.reserve()
    initial_output.store(ttl.block.fill(0, shape=(PNHt, vDHt)))

    query_block = query_dfb.wait()
    key_block = key_dfb.wait()
    transposed_key_output = transposed_key_dfb.reserve()
    transposed_key_output.store(ttl.transpose(key_block))
    transposed_key = transposed_key_dfb.wait()
    scores_output = scores_dfb.reserve()
    scores_output.store(query_block @ transposed_key)

    scores = scores_dfb.wait()
    scores_for_reduce_output = scores_for_reduce_dfb.reserve()
    scores_for_reduce_output.store(scores)
    scores_for_reduce = scores_for_reduce_dfb.wait()
    chunk_max_output = chunk_max_dfb.reserve()
    chunk_max_output.store(ttl.math.reduce_max(scores_for_reduce, dims=[1]))
    scores_for_exp_output = scores_for_exp_dfb.reserve()
    scores_for_exp_output.store(scores_for_reduce)

    previous_max = running_max_dfb.wait()
    chunk_max = chunk_max_dfb.wait()
    merged_max_for_alpha_output = merged_max_for_alpha_dfb.reserve()
    merged_max_for_alpha_output.store(ttl.math.max(previous_max, chunk_max))
    merged_max_for_alpha = merged_max_for_alpha_dfb.wait()
    alpha_for_sum_output = alpha_for_sum_dfb.reserve()
    alpha_for_sum_output.store(
        ttl.exp(ttl.sub(previous_max, merged_max_for_alpha) * SCALE)
    )
    merged_max_for_broadcast_output = merged_max_for_broadcast_dfb.reserve()
    merged_max_for_broadcast_output.store(merged_max_for_alpha)

    merged_max_for_broadcast = merged_max_for_broadcast_dfb.wait()
    max_broadcast_output = max_broadcast_dfb.reserve()
    max_broadcast_output.store(
        ttl.block.broadcast(
            merged_max_for_broadcast,
            dims=[1],
            shape=(PNHt, Sk_chunk_t),
        )
    )
    merged_max_for_state_output = merged_max_for_state_dfb.reserve()
    merged_max_for_state_output.store(merged_max_for_broadcast)

    merged_max_for_state = merged_max_for_state_dfb.wait()
    next_max = running_max_dfb.reserve()
    next_max.store(merged_max_for_state)

    scores_for_exp = scores_for_exp_dfb.wait()
    max_broadcast = max_broadcast_dfb.wait()
    exp_scores_for_reduce_output = exp_scores_for_reduce_dfb.reserve()
    exp_scores_for_reduce_output.store(
        ttl.exp(ttl.sub(scores_for_exp, max_broadcast) * SCALE)
    )
    exp_scores_for_reduce = exp_scores_for_reduce_dfb.wait()
    chunk_sum_output = chunk_sum_dfb.reserve()
    chunk_sum_output.store(ttl.math.reduce_sum(exp_scores_for_reduce, dims=[1]))
    exp_scores_for_matmul_output = exp_scores_for_matmul_dfb.reserve()
    exp_scores_for_matmul_output.store(exp_scores_for_reduce)

    alpha_for_sum = alpha_for_sum_dfb.wait()
    previous_sum = running_sum_dfb.wait()
    chunk_sum = chunk_sum_dfb.wait()
    next_sum = running_sum_dfb.reserve()
    next_sum.store(ttl.add(ttl.mul(alpha_for_sum, previous_sum), chunk_sum))
    alpha_for_broadcast_output = alpha_for_broadcast_dfb.reserve()
    alpha_for_broadcast_output.store(alpha_for_sum)

    alpha_for_broadcast = alpha_for_broadcast_dfb.wait()
    alpha_broadcast_output = alpha_broadcast_dfb.reserve()
    alpha_broadcast_output.store(
        ttl.block.broadcast(
            alpha_for_broadcast,
            dims=[1],
            shape=(PNHt, vDHt),
        )
    )
    alpha_broadcast = alpha_broadcast_dfb.wait()
    previous_output = running_output_dfb.wait()
    corrected_output = corrected_output_dfb.reserve()
    corrected_output.store(ttl.mul(alpha_broadcast, previous_output))

    exp_scores_for_matmul = exp_scores_for_matmul_dfb.wait()
    value_block = value_dfb.wait()
    partial_value_output = partial_value_dfb.reserve()
    partial_value_output.store(exp_scores_for_matmul @ value_block)

    corrected_output_block = corrected_output_dfb.wait()
    partial_value = partial_value_dfb.wait()
    next_output = running_output_dfb.reserve()
    next_output.store(ttl.add(corrected_output_block, partial_value))

    final_max = running_max_dfb.wait()
    local_max_output = local_max_dfb.reserve()
    local_max_output.store(final_max)
    final_sum = running_sum_dfb.wait()
    local_sum_output = local_sum_dfb.reserve()
    local_sum_output.store(final_sum)
    final_output = running_output_dfb.wait()
    local_output = local_output_dfb.reserve()
    local_output.store(final_output)

    node_x, _ = ttl.node(dims=2)
    query_destination = query_dfb.reserve()
    ttl.copy(query[0:PNHt, 0:DHt], query_destination)
    key_base = node_x * St_per_node
    key_destination = key_dfb.reserve()
    ttl.copy(
        key[key_base : key_base + Sk_chunk_t, 0:DHt],
        key_destination,
    )
    value_destination = value_dfb.reserve()
    ttl.copy(
        value[key_base : key_base + Sk_chunk_t, 0:vDHt],
        value_destination,
    )


@ttl.operation()
def _merge_softmax_state(
    left_max_dfb: ttl.DFB,
    left_sum_dfb: ttl.DFB,
    left_output_dfb: ttl.DFB,
    right_max_dfb: ttl.DFB,
    right_sum_dfb: ttl.DFB,
    right_output_dfb: ttl.DFB,
    result_max_dfb: ttl.DFB,
    result_sum_dfb: ttl.DFB,
    result_output_dfb: ttl.DFB,
    merged_max_dfb: ttl.DFB,
    left_scale_dfb: ttl.DFB,
    right_scale_dfb: ttl.DFB,
):
    """Combine two streaming-softmax states."""
    left_max = left_max_dfb.wait()
    left_sum = left_sum_dfb.wait()
    left_output = left_output_dfb.wait()
    right_max = right_max_dfb.wait()
    right_sum = right_sum_dfb.wait()
    right_output = right_output_dfb.wait()

    merged_max_output = merged_max_dfb.reserve()
    merged_max_output.store(ttl.math.max(left_max, right_max))
    merged_max = merged_max_dfb.wait()
    left_scale_output = left_scale_dfb.reserve()
    left_scale_output.store(ttl.exp(ttl.sub(left_max, merged_max) * SCALE))
    right_scale_output = right_scale_dfb.reserve()
    right_scale_output.store(ttl.exp(ttl.sub(right_max, merged_max) * SCALE))

    left_scale = left_scale_dfb.wait()
    right_scale = right_scale_dfb.wait()
    merged_sum = ttl.add(
        ttl.mul(left_scale, left_sum),
        ttl.mul(right_scale, right_sum),
    )
    left_scale_broadcast = ttl.block.broadcast(left_scale, dims=[1], shape=(PNHt, vDHt))
    right_scale_broadcast = ttl.block.broadcast(
        right_scale, dims=[1], shape=(PNHt, vDHt)
    )
    merged_output = ttl.add(
        ttl.mul(left_scale_broadcast, left_output),
        ttl.mul(right_scale_broadcast, right_output),
    )

    result_max = result_max_dfb.reserve()
    result_max.store(merged_max)
    result_sum = result_sum_dfb.reserve()
    result_sum.store(merged_sum)
    result_output = result_output_dfb.reserve()
    result_output.store(merged_output)


@ttl.operation()
def _transfer_state_component(
    pipe_net: ttl.PipeNet,
    send_dfb: ttl.DFB,
    received_dfb: ttl.DFB,
):
    pipe_net.if_src(lambda state_pipe: ttl.copy(send_dfb.wait(), state_pipe).wait())
    pipe_net.if_dst(
        lambda state_pipe: ttl.copy(state_pipe, received_dfb.reserve()).wait()
    )


@ttl.operation()
def _transfer_softmax_state(
    max_pipe_net: ttl.PipeNet,
    sum_pipe_net: ttl.PipeNet,
    output_pipe_net: ttl.PipeNet,
    send_max_dfb: ttl.DFB,
    send_sum_dfb: ttl.DFB,
    send_output_dfb: ttl.DFB,
    received_max_dfb: ttl.DFB,
    received_sum_dfb: ttl.DFB,
    received_output_dfb: ttl.DFB,
):
    """Transfer one streaming-softmax state over a tree level."""
    _transfer_state_component(max_pipe_net, send_max_dfb, received_max_dfb)
    _transfer_state_component(sum_pipe_net, send_sum_dfb, received_sum_dfb)
    _transfer_state_component(output_pipe_net, send_output_dfb, received_output_dfb)


@ttl.operation()
def flash_tree_reduce_8(
    local_max_dfb: ttl.DFB,
    local_sum_dfb: ttl.DFB,
    local_output_dfb: ttl.DFB,
    output,
):
    """Reduce eight streaming-softmax states and write the normalized result."""
    level_zero_max = ttl.PipeNet(
        [ttl.Pipe(src=(2 * index + 1, 0), dst=(2 * index, 0)) for index in range(4)]
    )
    level_zero_sum = ttl.PipeNet(
        [ttl.Pipe(src=(2 * index + 1, 0), dst=(2 * index, 0)) for index in range(4)]
    )
    level_zero_output = ttl.PipeNet(
        [ttl.Pipe(src=(2 * index + 1, 0), dst=(2 * index, 0)) for index in range(4)]
    )
    level_one_max = ttl.PipeNet(
        [ttl.Pipe(src=(4 * index + 2, 0), dst=(4 * index, 0)) for index in range(2)]
    )
    level_one_sum = ttl.PipeNet(
        [ttl.Pipe(src=(4 * index + 2, 0), dst=(4 * index, 0)) for index in range(2)]
    )
    level_one_output = ttl.PipeNet(
        [ttl.Pipe(src=(4 * index + 2, 0), dst=(4 * index, 0)) for index in range(2)]
    )
    level_two_max = ttl.PipeNet([ttl.Pipe(src=(4, 0), dst=(0, 0))])
    level_two_sum = ttl.PipeNet([ttl.Pipe(src=(4, 0), dst=(0, 0))])
    level_two_output = ttl.PipeNet([ttl.Pipe(src=(4, 0), dst=(0, 0))])

    send_max_dfb = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)
    send_sum_dfb = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)
    merged_max_dfb = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)
    left_scale_dfb = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)
    right_scale_dfb = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=2)
    send_output_dfb = ttl.make_dfb("bf16", shape=(PNHt, vDHt), block_count=2)
    normalized_output_dfb = ttl.make_dfb("bf16", shape=(PNHt, vDHt), block_count=2)
    received_max_dfb = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=3)
    received_sum_dfb = ttl.make_dfb("bf16", shape=(PNHt, 1), block_count=3)
    received_output_dfb = ttl.make_dfb("bf16", shape=(PNHt, vDHt), block_count=3)

    if level_zero_max.is_src():
        local_max = local_max_dfb.wait()
        local_sum = local_sum_dfb.wait()
        local_output = local_output_dfb.wait()
        send_max_output = send_max_dfb.reserve()
        send_max_output.store(local_max)
        send_sum_output = send_sum_dfb.reserve()
        send_sum_output.store(local_sum)
        send_output = send_output_dfb.reserve()
        send_output.store(local_output)

    if level_zero_max.is_dst():
        if level_one_max.is_dst():
            _merge_softmax_state(
                local_max_dfb,
                local_sum_dfb,
                local_output_dfb,
                received_max_dfb,
                received_sum_dfb,
                received_output_dfb,
                local_max_dfb,
                local_sum_dfb,
                local_output_dfb,
                merged_max_dfb,
                left_scale_dfb,
                right_scale_dfb,
            )
        if level_one_max.is_src():
            _merge_softmax_state(
                local_max_dfb,
                local_sum_dfb,
                local_output_dfb,
                received_max_dfb,
                received_sum_dfb,
                received_output_dfb,
                send_max_dfb,
                send_sum_dfb,
                send_output_dfb,
                merged_max_dfb,
                left_scale_dfb,
                right_scale_dfb,
            )

    if level_one_max.is_dst():
        if level_two_max.is_dst():
            _merge_softmax_state(
                local_max_dfb,
                local_sum_dfb,
                local_output_dfb,
                received_max_dfb,
                received_sum_dfb,
                received_output_dfb,
                local_max_dfb,
                local_sum_dfb,
                local_output_dfb,
                merged_max_dfb,
                left_scale_dfb,
                right_scale_dfb,
            )
        if level_two_max.is_src():
            _merge_softmax_state(
                local_max_dfb,
                local_sum_dfb,
                local_output_dfb,
                received_max_dfb,
                received_sum_dfb,
                received_output_dfb,
                send_max_dfb,
                send_sum_dfb,
                send_output_dfb,
                merged_max_dfb,
                left_scale_dfb,
                right_scale_dfb,
            )

    if level_two_max.is_dst():
        state_max = local_max_dfb.wait()
        state_sum = local_sum_dfb.wait()
        state_output = local_output_dfb.wait()
        received_max = received_max_dfb.wait()
        received_sum = received_sum_dfb.wait()
        received_output = received_output_dfb.wait()
        merged_max_output = merged_max_dfb.reserve()
        merged_max_output.store(ttl.math.max(state_max, received_max))
        merged_max = merged_max_dfb.wait()
        state_scale_output = left_scale_dfb.reserve()
        state_scale_output.store(ttl.exp(ttl.sub(state_max, merged_max) * SCALE))
        received_scale_output = right_scale_dfb.reserve()
        received_scale_output.store(ttl.exp(ttl.sub(received_max, merged_max) * SCALE))
        state_scale = left_scale_dfb.wait()
        received_scale = right_scale_dfb.wait()
        merged_sum = ttl.add(
            ttl.mul(state_scale, state_sum),
            ttl.mul(received_scale, received_sum),
        )
        state_scale_broadcast = ttl.block.broadcast(
            state_scale, dims=[1], shape=(PNHt, vDHt)
        )
        received_scale_broadcast = ttl.block.broadcast(
            received_scale, dims=[1], shape=(PNHt, vDHt)
        )
        merged_output = ttl.add(
            ttl.mul(state_scale_broadcast, state_output),
            ttl.mul(received_scale_broadcast, received_output),
        )
        normalization_sum_output = merged_max_dfb.reserve()
        normalization_sum_output.store(merged_sum)
        normalization_sum_scalar = merged_max_dfb.wait()
        normalized_output = normalized_output_dfb.reserve()
        normalization_sum = ttl.block.broadcast(
            normalization_sum_scalar,
            dims=[1],
            shape=(PNHt, vDHt),
        )
        normalized_output.store(
            ttl.mul(merged_output, ttl.math.recip(normalization_sum))
        )

    _transfer_softmax_state(
        level_zero_max,
        level_zero_sum,
        level_zero_output,
        send_max_dfb,
        send_sum_dfb,
        send_output_dfb,
        received_max_dfb,
        received_sum_dfb,
        received_output_dfb,
    )
    _transfer_softmax_state(
        level_one_max,
        level_one_sum,
        level_one_output,
        send_max_dfb,
        send_sum_dfb,
        send_output_dfb,
        received_max_dfb,
        received_sum_dfb,
        received_output_dfb,
    )
    _transfer_softmax_state(
        level_two_max,
        level_two_sum,
        level_two_output,
        send_max_dfb,
        send_sum_dfb,
        send_output_dfb,
        received_max_dfb,
        received_sum_dfb,
        received_output_dfb,
    )

    if level_two_max.is_dst():
        output_block = normalized_output_dfb.wait()
        ttl.copy(output_block, output[0:PNHt, 0:vDHt])


@ttl.operation(grid=(NNODES, 1), fp32_dest_acc_en=False)
def flash_chain_8node(query, key, value, output):
    """Compose local flash attention with an eight-node tree reduction."""
    local_max_dfb = ttl.make_dataflow_buffer_like(key, shape=(PNHt, 1), block_count=2)
    local_sum_dfb = ttl.make_dataflow_buffer_like(key, shape=(PNHt, 1), block_count=2)
    local_output_dfb = ttl.make_dataflow_buffer_like(
        key, shape=(PNHt, vDHt), block_count=2
    )

    flash_shard_kd(
        query,
        key,
        value,
        local_max_dfb,
        local_sum_dfb,
        local_output_dfb,
    )
    flash_tree_reduce_8(
        local_max_dfb,
        local_sum_dfb,
        local_output_dfb,
        output,
    )


@pytest.fixture(scope="module")
def flash_device():
    """Provide enough kernel-config L1 for the composed compute program."""
    if not is_hardware_available():
        pytest.skip("No Tenstorrent device available")

    max_worker_l1_size = ttnn.device.get_max_worker_l1_unreserved_size()
    worker_l1_size = max_worker_l1_size - KERNEL_CONFIG_BUFFER_RESERVE_BYTES
    device = ttnn.open_device(device_id=0, worker_l1_size=worker_l1_size)
    yield device
    ttnn.close_device(device)


def test_flash_chain_8node(flash_device):
    torch.manual_seed(2026)
    query_host = torch.randn(Q_ROWS, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    key_host = torch.randn(SEQ, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    value_host = torch.randn(SEQ, HEAD_DIM_V, dtype=torch.bfloat16) * 0.1
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            query_host.float().unsqueeze(0).unsqueeze(0),
            key_host.float().unsqueeze(0).unsqueeze(0),
            value_host.float().unsqueeze(0).unsqueeze(0),
            scale=SCALE,
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.bfloat16)
    )

    query = to_dram(query_host, flash_device)
    key = to_dram(key_host, flash_device)
    value = to_dram(value_host, flash_device)
    output = to_dram(
        torch.zeros(Q_ROWS, HEAD_DIM_V, dtype=torch.bfloat16),
        flash_device,
    )

    flash_chain_8node(query, key, value, output)

    actual = ttnn.to_torch(output).reshape(Q_ROWS, HEAD_DIM_V).float()
    assert_pcc(expected.float(), actual, threshold=PCC_THRESHOLD)
