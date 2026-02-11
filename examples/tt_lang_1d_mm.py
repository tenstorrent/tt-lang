# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from utils import assert_with_ulp
import ttl
from ttl import Program, copy, core, make_circular_buffer_like, Pipe, PipeNet
import matplotlib.pyplot as plt
import numpy as np


@ttl.kernel(grid='auto')
def matmul_1d(
    a_tensor: ttnn.Tensor, b_tensor: ttnn.Tensor, out_tensor: ttnn.Tensor, block_h: int, block_w: int, block_inner_dim: int, blocks_per_core_n: int
):
    assert a_tensor.shape[1] == b_tensor.shape[0], "Incompatible matrix shapes for multiplication."
    assert a_tensor.shape[0] == out_tensor.shape[0], "Output matrix has incorrect number of rows."
    assert b_tensor.shape[1] == out_tensor.shape[1], "Output matrix has incorrect number of columns."
    M = a_tensor.shape[0]
    N = b_tensor.shape[1]
    K = a_tensor.shape[1]
    Mt = M // ttnn.TILE_SIZE
    Kt = K // ttnn.TILE_SIZE
    Nt = N // ttnn.TILE_SIZE

    # tiling checks
    assert ttl.grid_size(dims=1) >= Nt // (blocks_per_core_n * block_w), "Not enough cores for the given tiling configuration."

    num_working_cores = Nt // (blocks_per_core_n * block_w)
    num_blocks_m = Mt // block_h
    num_blocks_k = Kt // block_inner_dim

    buffering_factor = 2
    a_cb = make_circular_buffer_like(
        a_tensor, shape=(block_h, block_inner_dim), buffer_factor=buffering_factor
    )
    b_cb = make_circular_buffer_like(
        b_tensor, shape=(block_inner_dim, block_w), buffer_factor=buffering_factor
    )
    # non buffered output, matching metal implementation
    out_cb = make_circular_buffer_like(
        out_tensor, shape=(block_h, block_w), buffer_factor=1
    )

    mcast_pipe = ttl.Pipe((0,), (slice(1, num_working_cores),))
    net = PipeNet([mcast_pipe])

    def block_slice(block_offset, block_size):
        return slice(block_offset * block_size, (block_offset + 1) * block_size)

    @ttl.compute()
    def mm_compute():
        core = ttl.core(dims=1)
        if core >= num_working_cores:
            return
        for block_m in range(num_blocks_m):
            for block_n in range(blocks_per_core_n):
                with out_cb.reserve() as out_blk:
                    for block_k in range(num_blocks_k):
                        with a_cb.wait() as a_blk, b_cb.wait() as b_blk:
                            print("Core", core, "computing block m:", block_m, "n:", (block_n + core * blocks_per_core_n), "k:", block_k)
                            out_blk.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def mm_A_reader_mcast():
        core = ttl.core(dims=1)
        if core >= num_working_cores:
            return
        for block_m in range(num_blocks_m):
            for _ in range(blocks_per_core_n):
                for block_k in range(num_blocks_k):
                    with a_cb.reserve() as a_blk:
                        def pipe_src(pipe):
                            in_rd = copy(
                                a_tensor[block_slice(block_m, block_h), block_slice(block_k, block_inner_dim)],
                                a_blk,
                            )
                            in_rd.wait()
                            mcast_wr = copy(a_blk, pipe)
                            mcast_wr.wait()
                            print("sent A block m:", block_m, "k:", block_k, "from core", core)

                        def pipe_dst(pipe):
                            mcast_rd = copy(pipe, a_blk)
                            mcast_rd.wait()
                            print("received A block m:", block_m, "k:", block_k, "on core", core)

                        net.if_src(pipe_src)
                        net.if_dst(pipe_dst)

    @ttl.datamovement()
    def mm_b_reader_out_writer():
        core = ttl.core(dims=1)
        if core >= num_working_cores:
            return
        for block_m in range(num_blocks_m):
            for block_n in range(blocks_per_core_n):
                for block_k in range(num_blocks_k):
                    with b_cb.reserve() as b_blk:
                        b_rd = copy(
                            b_tensor[block_slice(block_k, block_inner_dim), block_slice(block_n + core * blocks_per_core_n, block_w)],
                            b_blk,
                        )
                        b_rd.wait()
                with out_cb.wait() as out_blk:
                    out_wr = copy(
                        out_blk,
                        out_tensor[
                            block_slice(block_m, block_h),
                            block_slice(block_n + core * blocks_per_core_n, block_w),
                        ],
                    )
                    out_wr.wait()


def visualize_diff_heatmap(diff_tensor, test_name="", save_path=None):
    """
    Visualize the difference tensor as a heatmap.
    
    Args:
        diff_tensor: torch.Tensor containing the absolute differences
        test_name: string name for the test case
        save_path: optional path to save the image (e.g., 'diff_heatmap.png')
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to numpy for visualization (convert bfloat16 to float32 first)
    diff_np = diff_tensor.float().cpu().numpy()
    
    # Create heatmap
    im = plt.imshow(diff_np, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(im, label='Absolute Difference')
    plt.title(f'Difference Heatmap: {test_name}\nMax diff: {torch.max(diff_tensor):.6f}')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    # Add grid for better readability
    plt.grid(False)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()


def test_matmul_1d(M, N, K, block_h, block_w, block_inner_dim, blocks_per_core_n):

    A = ttnn.rand(
        (M, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    B = ttnn.rand(
        (K, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    output_t = ttnn.empty(
        (M, N),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    matmul_1d(A, B, output_t, block_h, block_w, block_inner_dim, blocks_per_core_n)

    golden_output = A.to_torch() @ B.to_torch()

    diff_tensor = torch.abs(output_t.to_torch() - golden_output)
    print(f"diff {diff_tensor}")
    print(f"max diff {torch.max(diff_tensor)}")
    
    # Visualize the diff tensor as a heatmap
    test_name = f"M={M}, N={N}, K={K}, block_h={block_h}, block_w={block_w}"
    visualize_diff_heatmap(diff_tensor, test_name=test_name, 
                          save_path=f"diff_heatmap_M{M}_N{N}_K{K}.png")

    assert_with_ulp(output_t.to_torch(), golden_output)

    print("Test passed!")

# M, N, K, block_h, block_w, block_inner_dim, blocks_per_core_n
# test_matmul_1d(32, 8*32, 32, 1, 1, 1, 1) # base, one row, single tile block, one block per core
# test_matmul_1d(64, 8*64, 64, 2, 2, 2, 1) # bigger than single tile blocks
# test_matmul_1d(32, 64*32, 32, 1, 1, 1, 1) # all wh cores
# test_matmul_1d(32, 8*32*2, 32, 1, 1, 1, 2) # multiple blocks per core (multiple blocks in n dim)
test_matmul_1d(64, 3*32, 32, 1, 1, 1, 1) # multiple blocks in m dim
#test_matmul_1d(32, 8*32, 64, 1, 1, 1, 1) # multiple blocks in k dim
# test_matmul_1d(64, 8*32*2, 64, 1, 1, 1, 2) # multiple blocks in all dims

