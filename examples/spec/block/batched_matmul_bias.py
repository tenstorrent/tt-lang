# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Example source for docs/sphinx/specs/TTLangSpecification.md.
#
# The lines between the "spec:begin" and "spec:end" markers below are included
# verbatim in the specification. Regenerate the specification after editing:
#
#     python docs/sphinx/specs/build_spec.py
#
# Everything outside the markers (imports, scaffolding) exists so the file can
# stand on its own and is not copied into the specification.

import math

import torch

import ttl
import ttnn

# spec:begin
# ---------------------
# Batched matrix multiplication with bias:
#
# y[l, m, n] = ∑(a[l, m, k] * b[k, n]) + c[m, n]
#              k
#
# Tensor   Torch shape   Note
# a        L, M, K       Batched a matrix (e.g. input activations)
# b        K, N          Non-batched b matrix (e.g. weights)
# c        M, N          Non-batched bias matrix c (e.g. weights)
# y        L, M, N       Batched y matrix (e.g. output activations)
#
# All tensors have tiled layout

# Shape in tiles (M, N and K are evenly divisible by TILE_SIZE)
M_TILES = M // TILE_SIZE
N_TILES = N // TILE_SIZE
K_TILES = K // TILE_SIZE

# Shape in blocks (L, M_TILES, N_TILES and K_TILES are evenly
# divisible by L_BLOCK_SIZE, M_BLOCK_SIZE, N_BLOCK_SIZE and K_BLOCK_SIZE)
L_BLOCKS = L // L_BLOCK_SIZE
M_BLOCKS = M_TILES // M_BLOCK_SIZE
N_BLOCKS = N_TILES // N_BLOCK_SIZE
K_BLOCKS = K_TILES // K_BLOCK_SIZE

a_dfb = ttl.make_dataflow_buffer_like(
    a, shape=(L_BLOCK_SIZE, M_BLOCK_SIZE, K_BLOCK_SIZE)
)
b_dfb = ttl.make_dataflow_buffer_like(b, shape=(K_BLOCK_SIZE, N_BLOCK_SIZE))
c_dfb = ttl.make_dataflow_buffer_like(c, shape=(M_BLOCK_SIZE, N_BLOCK_SIZE))
y_dfb = ttl.make_dataflow_buffer_like(
    y, shape=(L_BLOCK_SIZE, M_BLOCK_SIZE, N_BLOCK_SIZE)
)


@ttl.datamovement()
def matmul_read():
    for l_block in range(L_BLOCKS):
        l_slice = slice(l_block * L_BLOCK_SIZE, (l_block + 1) * L_BLOCK_SIZE)

        for m_block in range(M_BLOCKS):
            m_slice = slice(m_block * M_BLOCK_SIZE, (m_block + 1) * M_BLOCK_SIZE)

            for n_block in range(N_BLOCKS):
                n_slice = slice(n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE)

                # Reserve c_blk
                with c_dfb.reserve() as c_blk:

                    # Load M_BLOCK_SIZE×N_BLOCK_SIZE block of c into c_blk
                    c_xf = ttl.copy(c[m_slice, n_slice], c_blk)
                    c_xf.wait()

                    # End of "with" scope:
                    # Push c_blk to make it ready for matmul_compute

                # Repeat for each K block
                for k_block in range(K_BLOCKS):
                    k_slice = slice(
                        k_block * K_BLOCK_SIZE, (k_block + 1) * K_BLOCK_SIZE
                    )

                    # Reserve a_blk and b_blk
                    with (
                        a_dfb.reserve() as a_blk,
                        b_dfb.reserve() as b_blk,
                    ):
                        # Load L_BLOCK_SIZE×M_BLOCK_SIZE×K_BLOCK_SIZE of a into a_blk
                        # and K_BLOCK_SIZE×N_BLOCK_SIZE of b into b_blk
                        a_xf = ttl.copy(a[l_slice, m_slice, k_slice], a_blk)
                        b_xf = ttl.copy(b[k_slice, n_slice], b_blk)

                        a_xf.wait()
                        b_xf.wait()

                        # End of "with" scope:
                        # Push a_blk and b_blk to make it ready for matmul_compute


@ttl.compute()
def matmul_compute():
    for _ in range(L_BLOCKS):
        for _ in range(M_BLOCKS):
            for _ in range(N_BLOCKS):

                # Reserve y_blk
                with y_dfb.reserve() as y_blk:

                    # Zero-initialize the accumulator y_final before summing K_BLOCKS partial products
                    y_final = ttl.block.fill(
                        0, shape=(L_BLOCK_SIZE, M_BLOCK_SIZE, N_BLOCK_SIZE)
                    )

                    # Repeat for each K block
                    for _ in range(K_BLOCKS):

                        # Wait for a_blk and b_blk to be loaded and pushed by matmul_read
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                        ):
                            # b_blk has shape K_BLOCK_SIZE×N_BLOCK_SIZE;
                            # Unsqueeze it to 1×K_BLOCK_SIZE×N_BLOCK_SIZE and then
                            # broadcast it over dim 0 to L_BLOCK_SIZE×K_BLOCK_SIZE×N_BLOCK_SIZE
                            b_bcast = ttl.block.broadcast(
                                ttl.block.unsqueeze(b_blk, dims=[0]),
                                dims=[0],
                                shape=(L_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE),
                            )

                            # Accumulate dot product between L_BLOCK_SIZE×M_BLOCK_SIZE×K_BLOCK_SIZE a_blk and
                            # L_BLOCK_SIZE×K_BLOCK_SIZE×N_BLOCK_SIZE b_bcast in y_final
                            y_final += a_blk @ b_bcast

                            # End of "with" scope:
                            # Pop a_blk and b_blk to make them available for matmul_read to load and push next blocks

                    # Wait for c_blk to be loaded and pushed by matmul_read
                    with c_dfb.wait() as c_blk:

                        # c_blk has shape M_BLOCK_SIZE×N_BLOCK_SIZE;
                        # Unsqueeze it to 1×M_BLOCK_SIZE×N_BLOCK_SIZE and then
                        # broadcast it over dim 0 to L_BLOCK_SIZE×M_BLOCK_SIZE×N_BLOCK_SIZE
                        c_bcast = ttl.block.broadcast(
                            ttl.block.unsqueeze(c_blk, dims=[0]),
                            dims=[0],
                            shape=(L_BLOCK_SIZE, M_BLOCK_SIZE, N_BLOCK_SIZE),
                        )

                        y_final = y_final + c_bcast

                        # End of "with" scope:
                        # Pop c_blk to make it available for matmul_read to load and push next block

                    y_blk.store(y_final)

                    # End of "with" scope:
                    # Push y_blk to make it ready for matmul_write


@ttl.datamovement()
def matmul_write():
    for l_block in range(L_BLOCKS):
        for m_block in range(M_BLOCKS):
            for n_block in range(N_BLOCKS):

                # Wait for matmul_compute to store and push y_blk
                with y_dfb.wait() as y_blk:

                    # Store L_BLOCK_SIZE×M_BLOCK_SIZE×N_BLOCK_SIZE y_blk block into y
                    y_xf = ttl.copy(
                        y_blk,
                        y[
                            l_block * L_BLOCK_SIZE : (l_block + 1) * L_BLOCK_SIZE,
                            m_block * M_BLOCK_SIZE : (m_block + 1) * M_BLOCK_SIZE,
                            n_block * N_BLOCK_SIZE : (n_block + 1) * N_BLOCK_SIZE,
                        ],
                    )
                    y_xf.wait()

                    # End of "with" scope:
                    # Pop y_blk to make it available for matmul_compute to store and push next block


# spec:end
