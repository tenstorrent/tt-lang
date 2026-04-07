# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import ttnn
import ttl

from utils.correctness import assert_with_ulp
from utils.block_allocation import get_number_of_nodes_from_ranges, split_work_to_nodes


@ttl.operation(grid=("auto"))
def tt_lang_multinode_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."
    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_output_tiles_total = (M * N) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)
    dfb_block_count = 2
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=dfb_block_count)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=dfb_block_count)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), block_count=dfb_block_count
    )

    print(f"num_output_tiles_total: {num_output_tiles_total}")
    all_nodes, node_group_1, node_group_2, work_per_node1, work_per_node2 = (
        split_work_to_nodes(
            (ttl.grid_size(dims=1),), num_output_tiles_total, row_wise=True
        )
    )
    print(
        f"all_nodes: {all_nodes}, node_group_1: {node_group_1}, node_group_2: {node_group_2}, work_per_node1: {work_per_node1}, work_per_node2: {work_per_node2}"
    )

    num_nodes_group_1 = get_number_of_nodes_from_ranges(node_group_1)
    num_nodes_group_2 = get_number_of_nodes_from_ranges(node_group_2)

    def get_tiles_per_node(node_id):
        if node_id < num_nodes_group_1:
            return work_per_node1
        elif node_id < num_nodes_group_1 + num_nodes_group_2:
            return work_per_node2
        else:  # no work assigned
            return 0

    def get_start_tile_id(node_id):
        if node_id < num_nodes_group_1:
            return node_id * work_per_node1
        elif node_id < num_nodes_group_1 + num_nodes_group_2:
            return (
                num_nodes_group_1 * work_per_node1
                + (node_id - num_nodes_group_1) * work_per_node2
            )
        else:  # no work assigned
            return 0

    @ttl.compute()
    def mm_compute():
        node_id = ttl.node(dims=1)
        for _ in range(get_tiles_per_node(node_id)):
            with out_dfb.reserve() as out_blk:
                acc = ttl.math.fill(out_blk, 0)
                for _ in range(Kt):
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                        acc += a_blk @ b_blk
                out_blk.store(acc)

    @ttl.datamovement()
    def mm_reader():
        node_id = ttl.node(dims=1)
        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_id in range(get_tiles_per_node(node_id)):
            current_tile_id = get_start_tile_id(node_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt
            for k in range(Kt):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    a_wr = ttl.copy(a[out_row, k], a_blk)
                    b_wr = ttl.copy(b[k, out_col], b_blk)
                    a_wr.wait()
                    b_wr.wait()

    @ttl.datamovement()
    def mm_writer():
        node_id = ttl.node(dims=1)
        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_id in range(get_tiles_per_node(node_id)):
            current_tile_id = get_start_tile_id(node_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt
            with out_dfb.wait() as out_blk:
                out_wr = ttl.copy(out_blk, out[out_row, out_col])
                out_wr.wait()


@pytest.mark.parametrize("M,K,N", [(640, 640, 640)])
def test_multinode_matmul_tt_lang(M, K, N):
    """Test multinode matmul operation."""
    device = ttnn.open_device(device_id=0)
    a = ttnn.rand((M, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.rand((K, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    c = ttnn.empty((M, N), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_lang_multinode_matmul(a, b, c)

    golden = torch.matmul(
        ttnn.to_torch(a).to(torch.bfloat16), ttnn.to_torch(b).to(torch.bfloat16)
    )
    result = ttnn.to_torch(c).to(torch.bfloat16)
    assert_with_ulp(golden, result)
    print("Test passed!")

    ttnn.close_device(device)


if __name__ == "__main__":
    test_multinode_matmul_tt_lang(256, 256, 256)
    test_multinode_matmul_tt_lang(512, 512, 512)
    test_multinode_matmul_tt_lang(640, 640, 640)
