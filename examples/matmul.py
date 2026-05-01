# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TTLANG_HARDWARE_CI: xfail-compiler

# Batched matrix multiplication with bias: y[i,m,n] = sum_k(a[i,m,k] * b[k,n]) + c[m,n]
#
# Tensor   Torch shape   Shape in tiles
# a        I, M, K       I, MT, KT
# b        K, N          KT, NT
# c        M, N          MT, NT
# y        I, M, N       I, MT, NT
#
# MT = M // TILE_SIZE, NT = N // TILE_SIZE, KT = K // TILE_SIZE
#
# The batch dimension I is iterated directly (one item per block).
# M, N and K are tile-divided into blocks of M_BLOCK_SIZE, N_BLOCK_SIZE
# and K_BLOCK_SIZE tiles each.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch

import ttl
import ttnn

TILE_SIZE = 32

# Block sizes in tiles for M, N and K dimensions
M_BLOCK_SIZE = 1
N_BLOCK_SIZE = 1
K_BLOCK_SIZE = 1


@ttl.operation(grid=(1, 1))
def matmul_with_bias(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:
    I = a.shape[0]
    M_TILES = a.shape[1] // TILE_SIZE
    K_TILES = a.shape[2] // TILE_SIZE
    N_TILES = b.shape[1] // TILE_SIZE

    M_BLOCKS = M_TILES // M_BLOCK_SIZE
    N_BLOCKS = N_TILES // N_BLOCK_SIZE
    K_BLOCKS = K_TILES // K_BLOCK_SIZE

    # Block shapes: the batch (I) dimension is one item per block
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, M_BLOCK_SIZE, K_BLOCK_SIZE))
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(K_BLOCK_SIZE, N_BLOCK_SIZE))
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(M_BLOCK_SIZE, N_BLOCK_SIZE))
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, M_BLOCK_SIZE, N_BLOCK_SIZE))

    @ttl.datamovement()
    def matmul_read():
        for i in range(I):
            for m_block in range(M_BLOCKS):
                m_slice = slice(m_block * M_BLOCK_SIZE, (m_block + 1) * M_BLOCK_SIZE)
                for n_block in range(N_BLOCKS):
                    n_slice = slice(
                        n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE
                    )

                    # Load M_BLOCK_SIZE x N_BLOCK_SIZE block of c
                    with c_dfb.reserve() as c_blk:
                        c_xf = ttl.copy(c[m_slice, n_slice], c_blk)
                        c_xf.wait()

                        # Push c_blk

                    for k_block in range(K_BLOCKS):
                        k_slice = slice(
                            k_block * K_BLOCK_SIZE, (k_block + 1) * K_BLOCK_SIZE
                        )

                        with (
                            a_dfb.reserve() as a_blk,
                            b_dfb.reserve() as b_blk,
                        ):
                            a_xf = ttl.copy(a[i, m_slice, k_slice], a_blk)
                            b_xf = ttl.copy(b[k_slice, n_slice], b_blk)
                            a_xf.wait()
                            b_xf.wait()

                            # Push a_blk and b_blk

    @ttl.compute()
    def matmul_compute():
        for _ in range(I):
            for _ in range(M_BLOCKS):
                for _ in range(N_BLOCKS):
                    with y_dfb.reserve() as y_blk:

                        # Zero-initialize y accumulator
                        y = ttl.block.fill(0, shape=(1, M_BLOCK_SIZE, N_BLOCK_SIZE))

                        for _ in range(K_BLOCKS):
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                            ):
                                # b_blk has shape (K_BLOCK_SIZE, N_BLOCK_SIZE);
                                # unsqueeze to (1, K_BLOCK_SIZE, N_BLOCK_SIZE) then
                                # broadcast to (1, K_BLOCK_SIZE, N_BLOCK_SIZE)
                                b = ttl.block.broadcast(
                                    ttl.block.unsqueeze(b_blk, dims=[0]),
                                    dims=[0],
                                    shape=(1, K_BLOCK_SIZE, N_BLOCK_SIZE),
                                )
                                y += a_blk @ b

                                # Pop a_blk and b_blk

                        with c_dfb.wait() as c_blk:
                            # c_blk has shape (M_BLOCK_SIZE, N_BLOCK_SIZE);
                            # unsqueeze to (1, M_BLOCK_SIZE, N_BLOCK_SIZE) then
                            # broadcast to (1, M_BLOCK_SIZE, N_BLOCK_SIZE)
                            c = ttl.block.broadcast(
                                ttl.block.unsqueeze(c_blk, dims=[0]),
                                dims=[0],
                                shape=(1, M_BLOCK_SIZE, N_BLOCK_SIZE),
                            )
                            y = y + c

                            # Pop c_blk

                        y_blk.store(y)

                        # Push y_blk

    @ttl.datamovement()
    def matmul_write():
        for i in range(I):
            for m_block in range(M_BLOCKS):
                m_slice = slice(m_block * M_BLOCK_SIZE, (m_block + 1) * M_BLOCK_SIZE)
                for n_block in range(N_BLOCKS):
                    n_slice = slice(
                        n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE
                    )

                    with y_dfb.wait() as y_blk:
                        y_xf = ttl.copy(y_blk, y[i, m_slice, n_slice])
                        y_xf.wait()

                        # Pop y_blk


def main() -> None:
    # I must be small since the batch dimension is iterated directly.
    I, M, K, N = 2, 64, 96, 128

    a_torch = torch.rand((I, M, K), dtype=torch.float32)
    b_torch = torch.rand((K, N), dtype=torch.float32)
    c_torch = torch.rand((M, N), dtype=torch.float32)

    a_in = ttnn.from_torch(a_torch)
    b_in = ttnn.from_torch(b_torch)
    c_in = ttnn.from_torch(c_torch)
    y_out = ttnn.empty((I, M, N), dtype=torch.float32)

    matmul_with_bias(a_in, b_in, c_in, y_out)

    result = ttnn.to_torch(y_out)
    expected = torch.stack([a_torch[i] @ b_torch + c_torch for i in range(I)])

    assert torch.allclose(result, expected, atol=1e-4), "Mismatch!"
    print("PASSED!")


if __name__ == "__main__":
    main()
