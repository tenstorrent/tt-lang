# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn

import ttl
from utils.block_allocation import get_large_matmul_params
from utils.correctness import assert_with_ulp


@ttl.kernel(grid=(13, 10))
def tt_lang_multinode_reuse_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    assert b.shape[1] == out.shape[1], "Output matrix has incorrect number of columns."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    K_block_size = 2  # k dim block size

    num_nodes_x, num_nodes_y = ttl.grid_size(dims=2)
    # unused subblock sizes, to be determined by compiler, but using helper function to get better simultaneous comparisons
    block_params = get_large_matmul_params(
        Mt, Nt, num_nodes_y, num_nodes_x, K_block_size
    )
    per_node_M = block_params.block_h
    per_node_N = block_params.block_w
    assert per_node_M != 0, "get_large_matmul_params was not able to find a solution"
    print(f"per_node_M: {per_node_M}, per_node_N: {per_node_N}")
    assert Mt % per_node_M == 0, "per_node_M must divide Mt"
    assert Nt % per_node_N == 0, "per_node_N must divide Nt"
    assert Kt % K_block_size == 0, "K_block_size must divide Kt"
    num_blocks_y = Mt // per_node_M
    num_blocks_x = Nt // per_node_N
    assert (
        num_blocks_x <= num_nodes_x and num_blocks_y <= num_nodes_y
    ), "number of total blocks must be less than or equal to num nodes"

    buffering_factor = 2
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(per_node_M, K_block_size), buffer_factor=buffering_factor
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(K_block_size, per_node_N), buffer_factor=buffering_factor
    )
    # non buffered output, matching metal implementation
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(per_node_M, per_node_N), buffer_factor=1
    )

    @ttl.compute()
    def mm_compute():
        node_x, node_y = ttl.node(dims=2)
        out_row = per_node_M * node_y
        out_col = per_node_N * node_x
        if (out_row < Mt) and (out_col < Nt):
            with out_dfb.reserve() as out_blk:  # per_node_M * per_node_N
                for _ in range(Kt // K_block_size):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                    ):  # a per_node_M x K_block_size, b K_block_size x per_node_N
                        out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        node_x, node_y = ttl.node(dims=2)
        out_row = per_node_M * node_y
        out_col = per_node_N * node_x
        if (out_row < Mt) and (out_col < Nt):
            for block in range(Kt // K_block_size):
                k = block * K_block_size
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    a_wr = ttl.copy(
                        a[out_row : (out_row + per_node_M), k : (k + K_block_size)],
                        a_blk,
                    )
                    b_wr = ttl.copy(
                        b[k : (k + K_block_size), out_col : (out_col + per_node_N)],
                        b_blk,
                    )
                    a_wr.wait()
                    b_wr.wait()

    # blocking only occurs on the k dim, so each node writes its entire output block at once
    @ttl.datamovement()
    def mm_writer():
        node_x, node_y = ttl.node(dims=2)
        out_row = per_node_M * node_y
        out_col = per_node_N * node_x
        if (out_row < Mt) and (out_col < Nt):
            with out_dfb.wait() as out_blk:
                out_wr = ttl.copy(
                    out_blk,
                    out[
                        out_row : (out_row + per_node_M),
                        out_col : (out_col + per_node_N),
                    ],
                )
                out_wr.wait()


@pytest.mark.parametrize("M,K,N", [(640, 640, 640)])
def test_multinode_reuse_matmul_tt_lang(M, K, N):
    """Test multinode matmul kernel."""
    device = ttnn.open_device(device_id=0)
    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_lang_multinode_reuse_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)

    ttnn.close_device(device)


@ttl.kernel(grid=(13, 10))
def tt_lang_multinode_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    num_nodes_x, num_nodes_y = ttl.grid_size(dims=2)
    # this simplified non-reuse multinode matmul is limited to 1 tile per node, to highlight differences with the reuse version
    assert num_nodes_x >= Nt
    assert num_nodes_y >= Mt

    buffering_factor = 2
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(1, 1), buffer_factor=buffering_factor
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(1, 1), buffer_factor=buffering_factor
    )
    # non buffered output, matching metal implementation
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def mm_compute():
        node_x, node_y = ttl.node(dims=2)
        out_row = node_y
        out_col = node_x
        if (out_row < Mt) and (out_col < Nt):
            with out_dfb.reserve() as out_blk:
                for _ in range(Kt):
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                        out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_reader():
        node_x, node_y = ttl.node(dims=2)
        out_row = node_y
        out_col = node_x
        if (out_row < Mt) and (out_col < Nt):
            for k in range(Kt):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    a_wr = ttl.copy(a[out_row, k], a_blk)
                    b_wr = ttl.copy(b[k, out_col], b_blk)
                    a_wr.wait()
                    b_wr.wait()

    # blocking only occurs on the k dim, so each node writes its entire output block at once
    @ttl.datamovement()
    def mm_writer():
        node_x, node_y = ttl.node(dims=2)
        out_row = node_y
        out_col = node_x
        if (out_row < Mt) and (out_col < Nt):
            with out_dfb.wait() as out_blk:
                out_wr = ttl.copy(
                    out_blk,
                    out[out_row, out_col],
                )
                out_wr.wait()
