# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused SDPA-style PipeNet test.

This covers a one-dispatch matmul-softmax-matmul chain where the first matmul
uses PipeNet SUMMA-style multicasts, the block-softmax stays in local DFBs, and
the second matmul consumes gathered softmax blocks plus multicast weights.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import is_hardware_available, to_dram
from utils.correctness import assert_allclose

TILE = 32
M_DIM = 512
K1_DIM = 512
N1_DIM = 512
N2_DIM = 512
BLOCK_M = 4
BLOCK_N = 4
BLOCK_K = 4
M_PARTS = 4
N_PARTS = 4
K_BLOCKS = K1_DIM // (BLOCK_K * TILE)
N1_BLOCKS = N1_DIM // (BLOCK_N * TILE)
WORKER_L1_SIZE = 1448000

assert N1_BLOCKS == N_PARTS


@pytest.fixture
def sdpa_device():
    if not is_hardware_available():
        pytest.skip("No Tenstorrent device available")

    device = ttnn.open_device(device_id=0, worker_l1_size=WORKER_L1_SIZE)
    yield device
    ttnn.close_device(device)


@ttl.operation(grid=(N_PARTS, M_PARTS), fp32_dest_acc_en=False)
def sdpa_fused(activations, weights_1, weights_2, output):
    activation_cb = ttl.make_dataflow_buffer_like(
        activations, shape=(BLOCK_M, BLOCK_K), block_count=2
    )
    weights_1_cb = ttl.make_dataflow_buffer_like(
        weights_1, shape=(BLOCK_K, BLOCK_N), block_count=2
    )
    activation_tmp_cb = ttl.make_dataflow_buffer_like(
        activations, shape=(BLOCK_M, BLOCK_K), block_count=2
    )
    weights_1_tmp_cb = ttl.make_dataflow_buffer_like(
        weights_1, shape=(BLOCK_K, BLOCK_N), block_count=2
    )
    score_cb = ttl.make_dataflow_buffer_like(
        activations, shape=(BLOCK_M, BLOCK_N), block_count=2
    )

    activation_net = ttl.PipeNet(
        [
            ttl.Pipe(src=(0, m_part), dst=(slice(0, N_PARTS), m_part))
            for m_part in range(M_PARTS)
        ]
    )
    weights_1_net = ttl.PipeNet(
        [
            ttl.Pipe(src=(n_part, 0), dst=(n_part, slice(0, M_PARTS)))
            for n_part in range(N_PARTS)
        ]
    )

    softmax_cb = ttl.make_dataflow_buffer_like(
        activations, shape=(BLOCK_M, BLOCK_N), block_count=2
    )

    softmax_recv_cb = ttl.make_dataflow_buffer_like(
        weights_2, shape=(BLOCK_M, BLOCK_N), block_count=N1_BLOCKS
    )
    weights_2_cb = ttl.make_dataflow_buffer_like(
        weights_2, shape=(BLOCK_N, BLOCK_N), block_count=2
    )
    weights_2_tmp_cb = ttl.make_dataflow_buffer_like(
        weights_2, shape=(BLOCK_N, BLOCK_N), block_count=2
    )
    output_cb = ttl.make_dataflow_buffer_like(
        output, shape=(BLOCK_M, BLOCK_N), block_count=2
    )

    softmax_gather_net = ttl.PipeNet(
        [
            ttl.Pipe(src=(src_col, m_part), dst=(slice(0, N_PARTS), m_part))
            for m_part in range(M_PARTS)
            for src_col in range(N_PARTS)
        ]
    )
    weights_2_net = ttl.PipeNet(
        [
            ttl.Pipe(src=(n_part, 0), dst=(n_part, slice(0, M_PARTS)))
            for n_part in range(N_PARTS)
        ]
    )

    @ttl.compute()
    def compute():
        score_block = score_cb.reserve()
        for k_block in range(K_BLOCKS):
            activation_block = activation_cb.wait()
            weights_1_block = weights_1_cb.wait()
            score_block += activation_block @ weights_1_block

        score_input = score_cb.wait()
        max_score = ttl.math.reduce_max(score_input, dims=[0, 1])
        shifted_score = ttl.sub(
            score_input,
            ttl.block.broadcast(max_score, dims=[0, 1], shape=score_input.shape),
        )
        exp_score = ttl.exp(shifted_score)
        sum_score = ttl.math.reduce_sum(exp_score, dims=[0, 1])
        inv_sum = ttl.recip(
            ttl.block.broadcast(sum_score, dims=[0, 1], shape=exp_score.shape)
        )
        softmax_cb.reserve().store(ttl.mul(exp_score, inv_sum))

        output_block = output_cb.reserve()
        for n1_block in range(N1_BLOCKS):
            softmax_block = softmax_recv_cb.wait()
            weights_2_block = weights_2_cb.wait()
            output_block += softmax_block @ weights_2_block

    @ttl.datamovement()
    def dm_brisc():
        node_col, node_row = ttl.node(dims=2)
        row_start = node_row * BLOCK_M
        col_start = node_col * BLOCK_N

        for k_block in range(K_BLOCKS):
            k_start = k_block * BLOCK_K

            def send_activation(pipe):
                tmp_write = activation_tmp_cb.reserve()
                ttl.copy(
                    activations[
                        row_start : row_start + BLOCK_M,
                        k_start : k_start + BLOCK_K,
                    ],
                    tmp_write,
                )
                tmp_read = activation_tmp_cb.wait()
                ttl.copy(tmp_read, pipe)

            activation_net.if_src(send_activation)

            def send_weights_1(pipe):
                tmp_write = weights_1_tmp_cb.reserve()
                ttl.copy(
                    weights_1[
                        k_start : k_start + BLOCK_K,
                        col_start : col_start + BLOCK_N,
                    ],
                    tmp_write,
                )
                tmp_read = weights_1_tmp_cb.wait()
                ttl.copy(tmp_read, pipe)

            weights_1_net.if_src(send_weights_1)

        local_softmax = softmax_cb.wait()

        def send_softmax(pipe):
            ttl.copy(local_softmax, pipe)

        softmax_gather_net.if_src(send_softmax)

        for n1_block in range(N1_BLOCKS):
            k_start = n1_block * BLOCK_N

            def send_weights_2(pipe):
                tmp_write = weights_2_tmp_cb.reserve()
                ttl.copy(
                    weights_2[
                        k_start : k_start + BLOCK_N,
                        col_start : col_start + BLOCK_N,
                    ],
                    tmp_write,
                )
                tmp_read = weights_2_tmp_cb.wait()
                ttl.copy(tmp_read, pipe)

            weights_2_net.if_src(send_weights_2)

    @ttl.datamovement()
    def dm_ncrisc():
        node_col, node_row = ttl.node(dims=2)
        row_start = node_row * BLOCK_M
        col_start = node_col * BLOCK_N

        for k_block in range(K_BLOCKS):

            def recv_activation(pipe):
                activation_block = activation_cb.reserve()
                ttl.copy(pipe, activation_block)

            activation_net.if_dst(recv_activation)

            def recv_weights_1(pipe):
                weights_1_block = weights_1_cb.reserve()
                ttl.copy(pipe, weights_1_block)

            weights_1_net.if_dst(recv_weights_1)

        def recv_softmax(pipe):
            softmax_slot = softmax_recv_cb.reserve()
            ttl.copy(pipe, softmax_slot)

        softmax_gather_net.if_dst(recv_softmax)

        for n1_block in range(N1_BLOCKS):

            def recv_weights_2(pipe):
                weights_2_block = weights_2_cb.reserve()
                ttl.copy(pipe, weights_2_block)

            weights_2_net.if_dst(recv_weights_2)

        output_block = output_cb.wait()
        ttl.copy(
            output_block,
            output[
                row_start : row_start + BLOCK_M,
                col_start : col_start + BLOCK_N,
            ],
        )


def torch_block_softmax(score_tensor):
    result = torch.empty_like(score_tensor)
    block_rows = BLOCK_M * TILE
    block_cols = BLOCK_N * TILE
    for row_block in range(score_tensor.shape[0] // block_rows):
        for col_block in range(score_tensor.shape[1] // block_cols):
            block = score_tensor[
                row_block * block_rows : (row_block + 1) * block_rows,
                col_block * block_cols : (col_block + 1) * block_cols,
            ]
            softmax = torch.softmax(block.to(torch.float32).flatten(), dim=-1)
            result[
                row_block * block_rows : (row_block + 1) * block_rows,
                col_block * block_cols : (col_block + 1) * block_cols,
            ] = softmax.reshape_as(block).to(torch.bfloat16)
    return result


@pytest.mark.requires_device
def test_sdpa_fused_pipenet(sdpa_device):
    torch.manual_seed(2026)

    activations_torch = torch.randn(M_DIM, K1_DIM, dtype=torch.bfloat16) * 0.02
    weights_1_torch = torch.randn(K1_DIM, N1_DIM, dtype=torch.bfloat16) * 0.15
    weights_2_torch = torch.randn(N1_DIM, N2_DIM, dtype=torch.bfloat16) * 0.03

    scores_ref = (activations_torch.float() @ weights_1_torch.float()).to(
        torch.bfloat16
    )
    softmax_ref = torch_block_softmax(scores_ref)
    output_ref = (softmax_ref.float() @ weights_2_torch.float()).to(torch.bfloat16)

    activations = to_dram(activations_torch, sdpa_device)
    weights_1 = to_dram(weights_1_torch, sdpa_device)
    weights_2 = to_dram(weights_2_torch, sdpa_device)
    output = to_dram(torch.zeros(M_DIM, N2_DIM, dtype=torch.bfloat16), sdpa_device)

    sdpa_fused(activations, weights_1, weights_2, output)
    ttnn.synchronize_device(sdpa_device)

    result = ttnn.to_torch(output).reshape(M_DIM, N2_DIM)
    assert_allclose(result.float(), output_ref.float(), rtol=1e-3, atol=1e-3)
