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

import math

import torch

import ttl
import ttnn

# Concrete compile-time sizes for a single-block run.
TILE_SIZE = 32
N, M = 64, 32
N_BLOCK_SIZE = 1


@ttl.operation(grid=(1, 1))
def elementwise_broadcast_reduce(
    a: ttnn.Tensor,  # input matrix a (N, M)
    b: ttnn.Tensor,  # column-wise vector b (N, 1)
    c: ttnn.Tensor,  # row-wise vector c (M,)
    d: ttnn.Tensor,  # scalar value d ()
    y: ttnn.Tensor,  # output vector y (N, 1)
    z: ttnn.Tensor,  # output vector z (M,)
) -> None:
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

    # Tiled DFB shape needs to be at least two-dimensional; when tiled, the vector b of
    # shape (N, 1) is placed in column 0 of each tile in a column of N_TILES tiles
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(N_BLOCK_SIZE, 1))
    # When tiled, the vector c of shape M is placed in row 0 of each tile in a row of M_TILES tiles
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, M_TILES))
    # When tiled, the scalar value d of shape () is placed at position (0, 0) of a single tile
    d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1))
    # When untiled, the vector y is formed from column 0 of each tile in a column of N_TILES tiles
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(N_BLOCK_SIZE, 1))
    # When untiled, the vector z is formed from row 0 of each tile in a row of M_TILES tiles
    z_dfb = ttl.make_dataflow_buffer_like(z, shape=(1, M_TILES))

    @ttl.datamovement()
    def elwise_read():

        # Reserve c_blk and d_blk blocks
        with (
            c_dfb.reserve() as c_blk,
            d_dfb.reserve() as d_blk,
        ):
            # Load entire (1×M_TILES) of c; when tiled, the vector c of shape M is placed
            # in row 0 of each tile in a row of M_TILES tiles
            c_xf = ttl.copy(c[0, :], c_blk)

            # Load entire (1×1) d; when tiled, the scalar value d of shape () is placed at
            # position (0, 0) of a single tile
            d_xf = ttl.copy(d[0, 0], d_blk)

            c_xf.wait()
            d_xf.wait()

            # End of "with" scope: c_blk and d_blk are pushed implicitly, which makes
            # them ready for elwise_compute

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

                # Load N_BLOCK_SIZE×1 block of b; when tiled, the vector b of shape (N, 1)
                # is placed in column 0 of each tile in a column of N_TILES tiles
                b_xf = ttl.copy(
                    b[n_block * N_BLOCK_SIZE : (n_block + 1) * N_BLOCK_SIZE, 0], b_blk
                )

                a_xf.wait()
                b_xf.wait()

                # End of "with" scope: a_blk and b_blk are pushed implicitly, which makes
                # them ready for elwise_compute

    @ttl.compute()
    def elwise_compute():

        # Wait for c_blk and d_blk to be loaded and pushed by elwise_read; reserve z_blk
        with (
            c_dfb.wait() as c_blk,
            d_dfb.wait() as d_blk,
            z_dfb.reserve() as z_blk,
        ):
            c_squared = c_blk**2
            d_squared = d_blk**2

            # Broadcast c_squared along dimension 0 (first) to get N_BLOCK_SIZE×M_TILES; this
            # first broadcasts column 0 to fill each of M_TILES tiles, then it broadcasts the
            # column of M_TILES tiles to get N_BLOCK_SIZE×M_TILES tiles
            c_squared_bcast = ttl.block.broadcast(
                c_squared, dims=[0], shape=(N_BLOCK_SIZE, M_TILES)
            )

            # Broadcast d_squared along all dimensions (0 and 1) to N_BLOCK_SIZE×M_TILES; this
            # first broadcasts the single scalar value at position (0, 0) to fill a single tile,
            # then it broadcasts that tile to get N_BLOCK_SIZE×M_TILES tiles
            d_squared_bcast = ttl.block.broadcast(
                d_squared, dims=[0, 1], shape=(N_BLOCK_SIZE, M_TILES)
            )

            # Zero-initialize the accumulator z before summing N_BLOCKS partial sums
            z_final = ttl.block.fill(0, shape=(1, M_TILES))

            for _ in range(N_BLOCKS):

                # Wait for a_blk and b_blk to be loaded and pushed by elwise_read; reserve y_blk
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    a_squared = a_blk**2
                    b_squared = b_blk**2

                    # Broadcast b_squared along dim -1 (last) to get N_BLOCK_SIZE×M_TILES; this
                    # first broadcasts row 0 to fill each of N_BLOCK_SIZE tiles, then it
                    # broadcasts the row of N_BLOCK_SIZE tiles to get N_BLOCK_SIZE×M_TILES tiles
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

                    # End of "with" scope: a_blk and b_blk are popped implicitly, which makes
                    # them available for elwise_read to load and push the next blocks, and
                    # y_blk is pushed implicitly, which makes it ready for elwise_write

            # Store z_final
            z_blk.store(z_final)

            # End of "with" scope: c_blk and d_blk are popped implicitly, and z_blk is
            # pushed implicitly, which makes it ready for elwise_write

    @ttl.datamovement()
    def elwise_write():

        # Wait for elwise_compute to store and push z_blk
        with z_dfb.wait() as z_blk:

            # Store entire (1xM_TILES) of z; when untiled, the vector z is formed from row 0
            # of each tile in a row of M_TILES tiles
            z_xf = ttl.copy(z_blk, z[0, :])
            z_xf.wait()

            # End of "with" scope: z_blk is popped implicitly

        for n_block in range(N_BLOCKS):
            n_slice = slice(n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE)

            # Wait for elwise_compute to store and push y_blk
            with y_dfb.wait() as y_blk:

                # Store N_BLOCK_SIZExM_TILES of y; when untiled, the vector y is formed from
                # column 0 of each tile in a column of N_TILES tiles
                y_xf = ttl.copy(y_blk, y[n_slice, :])
                y_xf.wait()

                # End of "with" scope: y_blk is popped implicitly, which makes it available
                # for elwise_compute to store and push the next block

    # spec:end


torch.manual_seed(0)

device = ttnn.open_device(device_id=0)

try:
    # Inputs are chosen so a dominates b, c and d: a^2 is at least 4 while the
    # three squares sum to less than 3, which keeps a^2 - b^2 - c^2 - d^2
    # strictly positive so the z branch never hits a negative square root.
    #
    # They are as large as that allows, because each of the three enters the
    # result through a broadcast this example exists to show: scaled down to a few
    # percent, each one moves the summed result by less than the comparison's
    # tolerance, and the goldens below would pass with any single broadcast
    # dropped from the math.
    a_torch = 2.0 + 0.5 * torch.rand(N, M, dtype=torch.float32)
    b_torch = 0.9 * torch.rand(N, 1, dtype=torch.float32)
    c_torch = 0.9 * torch.rand(M, dtype=torch.float32)
    d_torch = 0.9 * torch.rand((), dtype=torch.float32)

    a_t = ttnn.from_torch(a_torch, layout=ttnn.TILE_LAYOUT, device=device)
    b_t = ttnn.from_torch(b_torch, layout=ttnn.TILE_LAYOUT, device=device)
    c_t = ttnn.from_torch(c_torch, layout=ttnn.TILE_LAYOUT, device=device)
    d_t = ttnn.from_torch(d_torch, layout=ttnn.TILE_LAYOUT, device=device)
    y_t = ttnn.zeros(ttnn.Shape([N, 1]), layout=ttnn.TILE_LAYOUT, device=device)
    z_t = ttnn.zeros(ttnn.Shape([M]), layout=ttnn.TILE_LAYOUT, device=device)

    elementwise_broadcast_reduce(a_t, b_t, c_t, d_t, y_t, z_t)

    # Golden mirrors the reduce contract: reduce_sum collapses the requested
    # dimension all the way to a scalar per row/column (tile-grid reduce plus the
    # within-tile reduce), so y[n] sums over all m and z[m] sums over all n.
    a2 = a_torch**2
    b2 = b_torch**2  # (N, 1)
    c2 = c_torch**2  # (M,)
    d2 = d_torch**2  # scalar
    y_golden = torch.sqrt(a2 + b2 + c2[None, :] + d2).sum(dim=1)  # (N,)
    z_golden = torch.sqrt(a2 - b2 - c2[None, :] - d2).sum(dim=0)  # (M,)

    # to_torch un-pads, so each result comes back at its declared shape: y as
    # the (N, 1) column it was created as, which the goldens compare as (N,).
    y_result = ttnn.to_torch(y_t)[:, 0]
    z_result = ttnn.to_torch(z_t)

    # The tolerance covers a different summation order over 32 float32 terms, and
    # no more: anything looser stops distinguishing the broadcasts above from each
    # other.
    assert torch.allclose(
        y_golden, y_result, rtol=1e-4, atol=1e-4
    ), "elementwise broadcast+reduce y did not match torch reference"
    assert torch.allclose(
        z_golden, z_result, rtol=1e-4, atol=1e-4
    ), "elementwise broadcast+reduce z did not match torch reference"

finally:
    ttnn.close_device(device)
