# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import ttnn

import ttl
from utils.block_allocation import get_large_matmul_params
from utils.correctness import assert_with_ulp


@ttl.operation(grid=("auto"))
def tt_lang_2d_mcast_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    assert b.shape[1] == out.shape[1], "Output matrix has incorrect number of columns."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    K_block_size = 2

    num_nodes_x, num_nodes_y = ttl.grid_size(dims=2)
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
    assert (
        num_blocks_x >= 2 and num_blocks_y >= 2
    ), "2D mcast requires at least a 2x2 active node grid"

    num_active_x = num_blocks_x
    num_active_y = num_blocks_y
    num_blocks_k = Kt // K_block_size

    block_count = 2
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(per_node_M, K_block_size), block_count=block_count
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(K_block_size, per_node_N), block_count=block_count
    )
    # non buffered output, matching metal implementation
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(per_node_M, per_node_N), block_count=1
    )

    # A multicast: left column (x=0) reads from DRAM and multicasts rightward along each row
    a_pipes = [
        ttl.Pipe((0, y), (slice(1, num_active_x), y)) for y in range(num_active_y)
    ]
    a_mcast_net = ttl.PipeNet(a_pipes)

    # B multicast: top row (y=0) reads from DRAM and multicasts downward along each column
    b_pipes = [
        ttl.Pipe((x, 0), (x, slice(1, num_active_y))) for x in range(num_active_x)
    ]
    b_mcast_net = ttl.PipeNet(b_pipes)

    @ttl.compute()
    def mm_compute():
        node_x, node_y = ttl.node(dims=2)
        out_row = per_node_M * node_y
        out_col = per_node_N * node_x
        if (out_row < Mt) and (out_col < Nt):
            with out_dfb.reserve() as out_blk:
                acc = ttl.math.fill(out_blk, 0)
                for _ in range(num_blocks_k):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                    ):
                        acc += a_blk @ b_blk
                out_blk.store(acc)

    @ttl.datamovement()
    def mm_reader():
        node_x, node_y = ttl.node(dims=2)
        out_row = per_node_M * node_y
        out_col = per_node_N * node_x
        if (out_row < Mt) and (out_col < Nt):
            for block_k in range(num_blocks_k):
                k = block_k * K_block_size

                # A: left column reads from DRAM and multicasts, other columns receive
                with a_dfb.reserve() as a_blk:

                    def a_pipe_src(pipe):
                        in_rd = ttl.copy(
                            a[
                                out_row : (out_row + per_node_M),
                                k : (k + K_block_size),
                            ],
                            a_blk,
                        )
                        in_rd.wait()
                        mcast_wr = ttl.copy(a_blk, pipe)
                        mcast_wr.wait()

                    def a_pipe_dst(pipe):
                        mcast_rd = ttl.copy(pipe, a_blk)
                        mcast_rd.wait()

                    a_mcast_net.if_src(a_pipe_src)
                    a_mcast_net.if_dst(a_pipe_dst)

                # B: top row reads from DRAM and multicasts, other rows receive
                with b_dfb.reserve() as b_blk:

                    def b_pipe_src(pipe):
                        in_rd = ttl.copy(
                            b[
                                k : (k + K_block_size),
                                out_col : (out_col + per_node_N),
                            ],
                            b_blk,
                        )
                        in_rd.wait()
                        mcast_wr = ttl.copy(b_blk, pipe)
                        mcast_wr.wait()

                    def b_pipe_dst(pipe):
                        mcast_rd = ttl.copy(pipe, b_blk)
                        mcast_rd.wait()

                    b_mcast_net.if_src(b_pipe_src)
                    b_mcast_net.if_dst(b_pipe_dst)

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


@pytest.mark.parametrize("M,K,N", [(3584, 768, 3072)])
def test_2d_mcast_matmul_tt_lang(M, K, N):
    """Test 2D multicast matmul operation."""
    device = ttnn.open_device(device_id=0)
    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_lang_2d_mcast_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)
    print("Test passed!")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_2d_mcast_matmul_tt_lang(3584, 768, 3072)
