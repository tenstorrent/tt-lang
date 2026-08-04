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
# Everything outside the markers (imports, scaffolding, the @ttl.operation
# wrapper, device setup, and the correctness check) exists so the file can run
# standalone; it is not copied into the specification. The marked lines are
# nested inside @ttl.operation and dedented on render, so these mechanics add
# nothing to the rendered text.


import torch

import ttl
import ttnn

# Concrete compile-time sizes for a single-block run.
TILE_SIZE = 32
L, M, N, K = 1, 32, 32, 32
L_BLOCK_SIZE = 1
M_BLOCK_SIZE = 1
N_BLOCK_SIZE = 1
K_BLOCK_SIZE = 1


@ttl.operation(grid=(1, 1))
def batched_matmul_bias(
    a: ttnn.Tensor,  # input tensor a (L, M, K)
    b: ttnn.Tensor,  # input tensor b (K, N)
    c: ttnn.Tensor,  # input bias tensor c (M, N)
    y: ttnn.Tensor,  # output tensor y (L, M, N)
) -> None:
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
                    n_slice = slice(
                        n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE
                    )

                    # Reserve c_blk
                    with c_dfb.reserve() as c_blk:

                        # Load M_BLOCK_SIZE×N_BLOCK_SIZE block of c into c_blk
                        c_xf = ttl.copy(c[m_slice, n_slice], c_blk)
                        c_xf.wait()

                        # End of "with" scope: c_blk is pushed implicitly, which makes it
                        # ready for matmul_compute

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

                            # End of "with" scope: a_blk and b_blk are pushed implicitly,
                            # which makes them ready for matmul_compute

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

                                # End of "with" scope: a_blk and b_blk are popped implicitly,
                                # which makes them available for matmul_read to load and push
                                # the next blocks

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

                            # End of "with" scope: c_blk is popped implicitly, which makes it
                            # available for matmul_read to load and push the next block

                        y_blk.store(y_final)

                        # End of "with" scope: y_blk is pushed implicitly, which makes it
                        # ready for matmul_write

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

                        # End of "with" scope: y_blk is popped implicitly, which makes it
                        # available for matmul_compute to store and push the next block

    # spec:end


device = ttnn.open_device(device_id=0)

try:
    a_t = ttnn.rand(ttnn.Shape([L, M, K]), layout=ttnn.TILE_LAYOUT, device=device)
    b_t = ttnn.rand(ttnn.Shape([K, N]), layout=ttnn.TILE_LAYOUT, device=device)
    c_t = ttnn.rand(ttnn.Shape([M, N]), layout=ttnn.TILE_LAYOUT, device=device)
    y_t = ttnn.zeros(ttnn.Shape([L, M, N]), layout=ttnn.TILE_LAYOUT, device=device)

    batched_matmul_bias(a_t, b_t, c_t, y_t)

    golden = torch.matmul(ttnn.to_torch(a_t), ttnn.to_torch(b_t)) + ttnn.to_torch(c_t)
    assert torch.allclose(
        golden, ttnn.to_torch(y_t), rtol=1e-1, atol=1e-1
    ), "batched matmul + bias did not match torch reference"

finally:
    ttnn.close_device(device)
