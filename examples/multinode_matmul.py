# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

import torch
import ttl
import ttnn
from utils.correctness import assert_pcc


@ttl.operation(grid=(8, 7))
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

    dfb_block_count = 2
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=dfb_block_count)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=dfb_block_count)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), block_count=dfb_block_count
    )

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    num_nodes = grid_rows * grid_cols
    tiles_per_node = -(-num_output_tiles_total // num_nodes)

    @ttl.compute()
    def mm_compute():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col

        for tile_offset in range(tiles_per_node):
            current_tile_id = node_id * tiles_per_node + tile_offset
            if current_tile_id < num_output_tiles_total:
                with out_dfb.reserve() as out_blk:
                    acc = ttl.block.fill(0, shape=out_blk.shape)
                    for _ in range(Kt):
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                            acc += a_blk @ b_blk
                    out_blk.store(acc)

    @ttl.datamovement()
    def mm_reader():
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col

        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_offset in range(tiles_per_node):
            current_tile_id = node_id * tiles_per_node + tile_offset
            if current_tile_id < num_output_tiles_total:
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
        node_col, node_row = ttl.node(dims=2)
        node_id = node_row * grid_cols + node_col

        # A[Mt, Kt] @ B[Kt, Nt] = C[Mt, Nt]
        for tile_offset in range(tiles_per_node):
            current_tile_id = node_id * tiles_per_node + tile_offset
            if current_tile_id < num_output_tiles_total:
                out_row = current_tile_id // Nt
                out_col = current_tile_id % Nt

                with out_dfb.wait() as out_blk:
                    out_wr = ttl.copy(out_blk, out[out_row, out_col])
                    out_wr.wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        # Test with matrices that are multiples of tile size.
        M, K, N = 128, 256, 64

        a_torch = torch.rand((M, K), dtype=torch.bfloat16)
        b_torch = torch.rand((K, N), dtype=torch.bfloat16)
        out_torch = torch.zeros((M, N), dtype=torch.bfloat16)

        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print(f"Matrix multiplication: ({M}, {K}) @ ({K}, {N}) = ({M}, {N})")
        print(f"Tiles: A={M//32}x{K//32}, B={K//32}x{N//32}, Out={M//32}x{N//32}")
        print(f"Total output tiles: {(M//32) * (N//32)}")
        print("Grid: 8x7 = 56 nodes")

        tt_lang_multinode_matmul(a, b, out)

        golden = a_torch @ b_torch
        assert_pcc(golden.float(), ttnn.to_torch(out).float(), threshold=0.99)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
