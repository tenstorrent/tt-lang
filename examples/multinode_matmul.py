# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler
# type: ignore

import ttl
import ttnn
from utils.correctness import assert_with_ulp


def get_number_of_nodes(node_range_set):
    """Get total number of nodes in a NodeRangeSet.

    Args:
        node_range_set: A NodeRangeSet containing one or more NodeRange objects

    Returns:
        Total number of nodes across all ranges
    """
    total_nodes = 0
    for node_range in node_range_set.ranges():
        x_range = node_range.end.x - node_range.start.x + 1
        y_range = node_range.end.y - node_range.start.y + 1
        total_nodes += x_range * y_range
    return total_nodes


@ttl.operation(grid=(13, 10))
def tt_lang_multinode_matmul(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    assert a.shape[1] == b.shape[0], "Incompatible matrix shapes for multiplication."
    assert a.shape[0] == out.shape[0], "Output matrix has incorrect number of rows."

    M = a.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE
    num_output_tiles_total = (M * N) // (ttnn.TILE_SIZE * ttnn.TILE_SIZE)

    buffering_factor = 2
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(1, 1), buffer_factor=buffering_factor
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(1, 1), buffer_factor=buffering_factor
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), buffer_factor=buffering_factor
    )

    # Get grid size and compute work distribution
    y_size, x_size = ttl.grid_size(dims=2)
    node_grid = ttnn.CoreCoord(x_size, y_size)

    print(f"node_grid: {node_grid}, num_output_tiles_total: {num_output_tiles_total}")
    (
        _,
        all_nodes,
        node_group_1,
        node_group_2,
        work_per_node1,
        work_per_node2,
    ) = ttnn.split_work_to_cores(node_grid, num_output_tiles_total, row_wise=True)
    print(
        f"all_nodes: {all_nodes}, node_group_1: {node_group_1}, node_group_2: {node_group_2}, "
        f"work_per_node1: {work_per_node1}, work_per_node2: {work_per_node2}"
    )

    num_nodes_group_1 = get_number_of_nodes(node_group_1)
    num_nodes_group_2 = get_number_of_nodes(node_group_2)

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
        num_tiles = get_tiles_per_node(node_id)

        for _ in range(num_tiles):
            # Reserve output block once for the entire K accumulation
            # The reserved block is automatically initialized with zeros
            with out_dfb.reserve() as out_blk:
                # Accumulate over K dimension
                acc = ttl.math.fill(out_blk, 0)
                for _ in range(Kt):
                    with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                        acc += a_blk @ b_blk
                out_blk.store(acc)

    @ttl.datamovement()
    def mm_reader():
        node_id = ttl.node(dims=1)
        num_tiles = get_tiles_per_node(node_id)

        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_id in range(num_tiles):
            current_tile_id = get_start_tile_id(node_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt

            for k in range(Kt):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    # Note: Using integer notation for tile indexing
                    a_wr = ttl.copy(a[out_row, k], a_blk)
                    b_wr = ttl.copy(b[k, out_col], b_blk)
                    a_wr.wait()
                    b_wr.wait()

    @ttl.datamovement()
    def mm_writer():
        node_id = ttl.node(dims=1)
        num_tiles = get_tiles_per_node(node_id)

        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_id in range(num_tiles):
            current_tile_id = get_start_tile_id(node_id) + tile_id
            out_row = current_tile_id // Nt
            out_col = current_tile_id % Nt

            with out_dfb.wait() as out_blk:
                out_wr = ttl.copy(out_blk, out[out_row, out_col])
                out_wr.wait()


def main() -> None:
    # Test with matrices that are multiples of tile size
    M, K, N = 128, 256, 64
    a = ttnn.rand((M, K), dtype=ttnn.float32)
    b = ttnn.rand((K, N), dtype=ttnn.float32)
    out = ttnn.empty((M, N), dtype=ttnn.float32)

    print(f"Matrix multiplication: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
    print(f"Tiles: A={M//32}x{K//32}, B={K//32}x{N//32}, Out={M//32}x{N//32}")
    print(f"Total output tiles: {(M//32) * (N//32)}")
    print(f"Grid: 8x8 = 64 nodes")

    tt_lang_multinode_matmul(a, b, out)

    # Compute golden result
    golden = a @ b

    # Verify correctness with relaxed tolerance for matmul
    assert_with_ulp(ttnn.to_torch(golden), ttnn.to_torch(out), ulp_threshold=1000)


if __name__ == "__main__":
    main()
