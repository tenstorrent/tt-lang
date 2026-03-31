# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Matmul with bias: Y = A @ B + C
# M=128, K=96, N=128 (4x3x4 in 32x32 tiles).
# DFB blocks: A (2,1), B (1,2), C/acc/Y (2,2) — 2x2 output block = 4 tiles,
# fits in bf16 DST (capacity 8). Outer loops iterate MT/2 × NT/2 output blocks.
#
# K-accumulation via `prev + a @ b`, which lowers to copy_tile(prev) +
# matmul_block(DST += A*B). The add is eliminated; no intermediate CB needed.
#
# DM access pattern: for each output block (mi, ni), the reader loads bias
# C[mi, ni] first, then streams KT pairs of A[mi, kt] and B[kt, ni].

import torch
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid=(1, 1))
def matmul_with_bias(
    A: ttnn.Tensor,
    B: ttnn.Tensor,
    C: ttnn.Tensor,
    Y: ttnn.Tensor,
) -> None:
    m_tiles = A.shape[0] // TILE_SIZE
    k_tiles = A.shape[1] // TILE_SIZE
    n_tiles = B.shape[1] // TILE_SIZE

    # Block sizes: 2x1 for A, 1x2 for B, 2x2 for output/accumulator.
    # Output block (2x2 = 4 tiles) fits in bf16 DST (capacity 8).
    block_rows, block_cols = 2, 2

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(block_rows, 1))
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, block_cols))
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(block_rows, block_cols))
    # Compute-local CB for K-accumulation (DM threads do not touch it).
    # buffer_factor=2: read previous partial while writing new one.
    acc_dfb = ttl.make_dataflow_buffer_like(
        Y, shape=(block_rows, block_cols), buffer_factor=2
    )
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(block_rows, block_cols))

    @ttl.datamovement()
    def read():
        for m_ind in range(m_tiles // block_rows):
            for n_ind in range(n_tiles // block_cols):
                row = m_ind * block_rows
                col = n_ind * block_cols
                with c_dfb.reserve() as c_blk:
                    tx = ttl.copy(
                        C[row : row + block_rows, col : col + block_cols], c_blk
                    )
                    tx.wait()

                for k in range(k_tiles):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        tx_a = ttl.copy(A[row : row + block_rows, k : k + 1], a_blk)
                        tx_a.wait()
                        tx_b = ttl.copy(B[k : k + 1, col : col + block_cols], b_blk)
                        tx_b.wait()

    @ttl.compute()
    def compute():
        for _ in range(m_tiles // block_rows):
            for _ in range(n_tiles // block_cols):
                # Pre-load bias into accumulator.
                with c_dfb.wait() as c_blk:
                    with acc_dfb.reserve() as acc:
                        acc.store(c_blk)

                # K-loop: prev + a @ b lowers to
                # copy_tile(prev → DST) + matmul_block(DST += A*B).
                for _ in range(k_tiles):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        acc_dfb.wait() as prev,
                    ):
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + a_blk @ b_blk)

                # Move accumulator to output.
                with acc_dfb.wait() as acc_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(acc_blk)

    @ttl.datamovement()
    def write():
        for m_ind in range(m_tiles // block_rows):
            for n_ind in range(n_tiles // block_cols):
                row = m_ind * block_rows
                col = n_ind * block_cols
                with y_dfb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk, Y[row : row + block_rows, col : col + block_cols]
                    )
                    tx.wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        M, K, N = 128, 96, 128

        A_torch = torch.randn((M, K), dtype=torch.bfloat16)
        B_torch = torch.randn((K, N), dtype=torch.bfloat16)
        C_torch = torch.randn((M, N), dtype=torch.bfloat16)

        A = ttnn.from_torch(
            A_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        B = ttnn.from_torch(
            B_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        C = ttnn.from_torch(
            C_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        Y = ttnn.from_torch(
            torch.zeros((M, N), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        matmul_with_bias(A, B, C, Y)

        result = ttnn.to_torch(Y)
        expected = A_torch @ B_torch + C_torch

        pcc = torch.corrcoef(
            torch.stack([result.flatten().float(), expected.flatten().float()])
        )[0, 1].item()
        assert pcc > 0.99, (
            f"PCC {pcc:.6f} < 0.99 for matmul+bias. "
            f"Max diff: {(result - expected).abs().max().item()}"
        )
        print("PASSED!")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
