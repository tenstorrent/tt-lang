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
# Tiled element-wise with broadcast and reduce:
#
# y[n] = ∑(√(a[n, m]² + b[n]² + c[m]² + d²))
#        j
#
# z[m] = ∑(√(a[n, m]² - b[n]² - c[m]² - d²))
#        i
#
# Tensor   Torch shape   Note
# a        N, M          N >> M
# b        N, 1          Column-wise vector — broadcast to match a along M
# c        M             Row-wise vector — broadcast to match a along N
# d        ()            Scalar value — broadcast to match a along N and M
# y        N, 1
# z        M
#
# All tensors have tiled layout

# Shape in tiles (N and M are evenly divisible by TILE_SIZE)
N_TILES = N // TILE_SIZE
M_TILES = M // TILE_SIZE

# Shape in blocks (N_TILES is evenly divisible by N_BLOCK_SIZE)
N_BLOCKS = N_TILES // N_BLOCK_SIZE

a_dfb = ttl.make_dataflow_buffer_like(a, shape=(N_BLOCK_SIZE, M_TILES))

# Tiled DFB shape needs to be at least two-dimensional;
# When tiled, the vector b of shape (N, 1) is placed in column 0
# of each tile in a column of N_TILES tiles
b_dfb = ttl.make_dataflow_buffer_like(b, shape=(N_BLOCK_SIZE, 1))
# When tiled, the vector c of shape M is placed in row 0
# of each tile in a row of M_TILES tiles
c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, M_TILES))
# When tiled, the scalar value d of shape () is placed at position (0, 0)
# of a single tile
d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1))
# When untiled, the vector y is formed from column 0
# of each tile in a column of N_TILES tiles
y_dfb = ttl.make_dataflow_buffer_like(y, shape=(N_BLOCK_SIZE, 1))
# When untiled, the vector z is formed from row 0
# of each tile in a row of M_TILES tiles
z_dfb = ttl.make_dataflow_buffer_like(z, shape=(1, M_TILES))


@ttl.datamovement()
def elwise_read():

    # Reserve c_blk and d_blk blocks
    with (
        c_dfb.reserve() as c_blk,
        d_dfb.reserve() as d_blk,
    ):
        # Load entire (1×M_TILES) of c;
        # When tiled, the vector c of shape M is placed in row 0
        # of each tile in a row of M_TILES tiles
        c_xf = ttl.copy(c[0, :], c_blk)

        # Load entire (1×1) d;
        # When tiled, the scalar value d of shape () is placed at position (0, 0)
        # of a single tile
        d_xf = ttl.copy(d[0, 0], d_blk)

        c_xf.wait()
        d_xf.wait()

        # End of "with" scope:
        # Push c_blk and d_blk to make them ready for elwise_compute

    for n_block in range(N_BLOCKS):

        # Reserve a_blk and b_blk blocks
        with (
            a_dfb.reserve() as a_blk,
            b_dfb.reserve() as b_blk,
        ):
            # Load N_BLOCK_SIZE×M_TILES block of a
            a_xf = ttl.copy(
                a[n_block * N_BLOCK_SIZE : (n_block + 1) * N_BLOCK_SIZE, :], a_blk
            )

            # Load N_BLOCK_SIZE×1 block of b;
            # When tiled, the vector b of shape (N, 1) is placed in column 0
            # of each tile in a column of N_TILES tiles
            b_xf = ttl.copy(
                b[n_block * N_BLOCK_SIZE : (n_block + 1) * N_BLOCK_SIZE, 0], b_blk
            )

            a_xf.wait()
            b_xf.wait()

            # End of "with" scope:
            # Push a_blk and b_blk to make them ready for elwise_compute


@ttl.compute()
def elwise_compute():

    # Wait for c_blk and d_blk to be loaded and pushed by elwise_read;
    # Reserve z_blk
    with (
        c_dfb.wait() as c_blk,
        d_dfb.wait() as d_blk,
        z_dfb.reserve() as z_blk,
    ):
        c_squared = c_blk**2
        d_squared = d_blk**2

        # Broadcast c_squared along dimension 0 (first) to get N_BLOCK_SIZE×M_TILES;
        # This first broadcasts column 0 to fill each of M_TILES tiles
        # then it broadcasts column of M_TILES tiles to get N_BLOCK_SIZE×M_TILES tiles
        c_squared_bcast = ttl.block.broadcast(
            c_squared, dims=[0], shape=(N_BLOCK_SIZE, M_TILES)
        )

        # Broadcast d_squared along all dimensions (0 and 1) to N_BLOCK_SIZE×M_TILES;
        # This first broadcasts single scalar value at position (0, 0) to fill a single tile
        # then it broadcasts single tile to get N_BLOCK_SIZE×M_TILES tiles
        d_squared_bcast = ttl.block.broadcast(
            d_squared, dims=[0, 1], shape=(N_BLOCK_SIZE, M_TILES)
        )

        # Zero-initialize the accumulator z before summing N_BLOCKS partial sums
        z_final = ttl.block.fill(0, shape=(1, M_TILES))

        for _ in range(N_BLOCKS):

            # Wait for a_blk and b_blk to be loaded and pushed by elwise_read;
            # Reserve y_blk
            with (
                a_dfb.wait() as a_blk,
                b_dfb.wait() as b_blk,
                y_dfb.reserve() as y_blk,
            ):
                a_squared = a_blk**2
                b_squared = b_blk**2

                # Broadcast b_squared along dim -1 (last) to get N_BLOCK_SIZE×M_TILES;
                # This first broadcasts row 0 to fill each of N_BLOCK_SIZE tiles
                # then it broadcasts row of N_BLOCK_SIZE tiles to get N_BLOCK_SIZE×M_TILES tiles
                b_squared_bcast = ttl.block.broadcast(
                    b_squared, dims=[-1], shape=(N_BLOCK_SIZE, M_TILES)
                )

                # Perform elementwise math on N_BLOCK_SIZE×M_TILES tiles
                expanded_y = ttl.math.sqrt(
                    a_squared + b_squared_bcast + c_squared_bcast + d_squared_bcast
                )
                expanded_z = ttl.math.sqrt(
                    a_squared - b_squared_bcast - c_squared_bcast - d_squared_bcast
                )

                # Reduce expanded_y along dim -1 (last) to get N_BLOCK_SIZE×1 row of tiles
                y_final = ttl.math.reduce_sum(
                    expanded_y, dims=[-1], shape=(N_BLOCK_SIZE, 1)
                )

                # Reduce expanded_z along dim 0 (first) to get 1×M_TILES column of tiles;
                z_partial = ttl.math.reduce_sum(
                    expanded_z, dims=[0], shape=(1, M_TILES)
                )

                # Store y_final
                y_blk.store(y_final)

                # Accumulate-add partial z_final
                z_final += z_partial

                # End of "with" scope:
                # Pop a_blk and b_dfb to make them available for elwise_read to load and push next blocks;
                # Push y_blk to make it ready for elwise_write

        # Store z_final
        z_blk.store(z_final)

        # End of "with" scope:
        # Pop c_blk and d_blk;
        # Push z_blk to make it ready for elwise_write


@ttl.datamovement()
def elwise_write():

    # Wait for elwise_compute to store and push z_blk
    with z_dfb.wait() as z_blk:

        # Store entire (1xM_TILES) of z;
        # When untiled, the vector z is formed from row 0
        # of each tile in a row of M_TILES tiles
        z_xf = ttl.copy(z_blk, z[0, :])
        z_xf.wait()

        # End of "with" scope:
        # Pop z_blk

    for n_block in range(N_BLOCKS):
        n_slice = slice(n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE)

        # Wait for elwise_compute to store and push y_blk
        with y_dfb.wait() as y_blk:

            # Store N_BLOCK_SIZExM_TILES of y;
            # When untiled, the vector y is formed from column 0
            # of each tile in a column of N_TILES tiles
            y_xf = ttl.copy(y_blk, y[n_slice, :])
            y_xf.wait()

            # End of "with" scope:
            # Pop y_blk to make it available for elwise_compute to store and push next block


# spec:end
