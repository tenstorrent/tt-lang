# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TTLANG_HARDWARE_CI: xfail-compiler

# Tiled element-wise with broadcast and reduce example from TTLangSpecification.md.
#
# Computes:
#   y[n] = sum_m( sqrt(a[n,m]^2 + b[n]^2 + c[m]^2 + d^2) )
#   z[m] = sum_n( sqrt(a[n,m]^2 - b[n]^2 - c[m]^2 - d^2) )
#
# Tensor   Torch shape    Shape in tiles (TILE = 32)
# a        N, M           N_TILES, M_TILES
# b        N_TILES, T     N_TILES, 1      (column vector)
# c        T,   M         1,   M_TILES    (row vector)
# d        T,   T         1,   1          (scalar block)
# y        N_TILES, T     N_TILES, 1      (column result)
# z        T,   M         1,   M_TILES    (row result)
#
# T = TILE_SIZE = 32 (a full tile width)
#
# NOTE: To ensure the simulator's tile-grid-level reduce matches the
# element-level golden reference, each tile stores a uniform value
# (all 32x32 elements equal). This means reduce over M_TILES tiles
# equals summing M_TILES independent scalar values -- the same as
# summing all M elements when each tile holds one logical value.

import math

import torch

from sim import ttl, ttnn

TILE_SIZE = 32

# Dimensions (in tiles)
N_TILES = 4
M_TILES = 3
N_BLOCK_SIZE = 2  # process N in blocks of N_BLOCK_SIZE tiles

N_BLOCKS = N_TILES // N_BLOCK_SIZE


def make_uniform_tile_tensor(tile_values: torch.Tensor) -> torch.Tensor:
    """Build a tensor where each logical tile holds one uniform scalar.

    Args:
        tile_values: Shape (..., rows, cols) of per-tile scalar values.

    Returns:
        Tensor of shape (..., rows*TILE_SIZE, cols*TILE_SIZE) where each
        tile block is filled with the corresponding scalar.
    """
    # tile_values[..., i, j] becomes a TILE_SIZE x TILE_SIZE block
    *batch, rows, cols = tile_values.shape
    out = tile_values[..., :, None, :, None]
    out = out.expand(*batch, rows, TILE_SIZE, cols, TILE_SIZE)
    return out.reshape(*batch, rows * TILE_SIZE, cols * TILE_SIZE).contiguous()


@ttl.operation(grid=(1, 1))
def eltwise_broadcast_reduce(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    d: ttnn.Tensor,
    y: ttnn.Tensor,
    z: ttnn.Tensor,
) -> None:
    # Shape in tiles
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(N_BLOCK_SIZE, M_TILES))
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(N_BLOCK_SIZE, 1))
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, M_TILES))
    d_dfb = ttl.make_dataflow_buffer_like(d, shape=(1, 1))
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(N_BLOCK_SIZE, 1))
    z_dfb = ttl.make_dataflow_buffer_like(z, shape=(1, M_TILES))

    @ttl.datamovement()
    def elwise_read():
        # Load c and d once (invariant across N blocks)
        with (
            c_dfb.reserve() as c_blk,
            d_dfb.reserve() as d_blk,
        ):
            c_xf = ttl.copy(c[0:1, 0:M_TILES], c_blk)
            d_xf = ttl.copy(d[0:1, 0:1], d_blk)
            c_xf.wait()
            d_xf.wait()

            # Push c_blk and d_blk

        for n_block in range(N_BLOCKS):
            n_slice = slice(n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE)

            with (
                a_dfb.reserve() as a_blk,
                b_dfb.reserve() as b_blk,
            ):
                a_xf = ttl.copy(a[n_slice, 0:M_TILES], a_blk)
                b_xf = ttl.copy(b[n_slice, 0:1], b_blk)
                a_xf.wait()
                b_xf.wait()

                # Push a_blk and b_blk

    @ttl.compute()
    def elwise_compute():
        # Wait for c and d; reserve z accumulator
        with (
            c_dfb.wait() as c_blk,
            d_dfb.wait() as d_blk,
            z_dfb.reserve() as z_blk,
        ):
            c_squared = c_blk**2
            d_squared = d_blk**2

            # Broadcast c_squared from (1, M_TILES) to (N_BLOCK_SIZE, M_TILES)
            c_squared_bcast = ttl.block.broadcast(
                c_squared, dims=[0], shape=(N_BLOCK_SIZE, M_TILES)
            )

            # Broadcast d_squared from (1, 1) to (N_BLOCK_SIZE, M_TILES)
            d_squared_bcast = ttl.block.broadcast(
                d_squared, dims=[0, 1], shape=(N_BLOCK_SIZE, M_TILES)
            )

            # Zero-initialize z accumulator
            z = ttl.block.fill(0, shape=(1, M_TILES))

            for _ in range(N_BLOCKS):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    a_squared = a_blk**2
                    b_squared = b_blk**2

                    # Broadcast b_squared from (N_BLOCK_SIZE, 1) to (N_BLOCK_SIZE, M_TILES)
                    b_squared_bcast = ttl.block.broadcast(
                        b_squared, dims=[-1], shape=(N_BLOCK_SIZE, M_TILES)
                    )

                    expanded_y = ttl.math.sqrt(
                        a_squared + b_squared_bcast + c_squared_bcast + d_squared_bcast
                    )
                    expanded_z = ttl.math.sqrt(
                        a_squared - b_squared_bcast - c_squared_bcast - d_squared_bcast
                    )

                    # Reduce expanded_y along M_TILES (last dim) -> (N_BLOCK_SIZE, 1)
                    y = ttl.math.reduce_sum(
                        expanded_y, dims=[-1], shape=(N_BLOCK_SIZE, 1)
                    )

                    # Partial reduce expanded_z along N_BLOCK_SIZE (first dim) -> (1, M_TILES)
                    z_partial = ttl.math.reduce_sum(
                        expanded_z, dims=[0], shape=(1, M_TILES)
                    )

                    y_blk.store(y)
                    z += z_partial

                    # Pop a_blk and b_blk; push y_blk

            z_blk.store(z)

            # Pop c_blk and d_blk; push z_blk

    @ttl.datamovement()
    def elwise_write():
        # Write z (single block covering all M_TILES)
        with z_dfb.wait() as z_blk:
            z_xf = ttl.copy(z_blk, z[0:1, 0:M_TILES])
            z_xf.wait()

        for n_block in range(N_BLOCKS):
            n_slice = slice(n_block * N_BLOCK_SIZE, (n_block + 1) * N_BLOCK_SIZE)
            with y_dfb.wait() as y_blk:
                y_xf = ttl.copy(y_blk, y[n_slice, 0:1])
                y_xf.wait()


def main() -> None:
    # Per-tile scalar values (each tile's 32x32 elements all share one value)
    a_vals = torch.rand((N_TILES, M_TILES), dtype=torch.float32) * 5.0 + 5.0
    b_vals = torch.rand((N_TILES,), dtype=torch.float32) * 0.5
    c_vals = torch.rand((M_TILES,), dtype=torch.float32) * 0.5
    d_val = torch.tensor(0.1, dtype=torch.float32)

    # Build uniform-tile tensors
    a_torch = make_uniform_tile_tensor(a_vals)
    b_torch = (
        make_uniform_tile_tensor(b_vals.view(N_TILES, 1))
        .expand(N_TILES * TILE_SIZE, TILE_SIZE)
        .contiguous()
    )
    c_torch = (
        make_uniform_tile_tensor(c_vals.view(1, M_TILES))
        .expand(TILE_SIZE, M_TILES * TILE_SIZE)
        .contiguous()
    )
    d_torch = d_val.expand(TILE_SIZE, TILE_SIZE).contiguous()

    # Create ttnn tensors
    a_in = ttnn.from_torch(a_torch)
    b_in = ttnn.from_torch(b_torch)
    c_in = ttnn.from_torch(c_torch)
    d_in = ttnn.from_torch(d_torch)
    y_out = ttnn.empty((N_TILES * TILE_SIZE, TILE_SIZE), dtype=torch.float32)
    z_out = ttnn.empty((TILE_SIZE, M_TILES * TILE_SIZE), dtype=torch.float32)

    eltwise_broadcast_reduce(a_in, b_in, c_in, d_in, y_out, z_out)

    y_result = ttnn.to_torch(y_out)
    z_result = ttnn.to_torch(z_out)

    # Golden reference: operate at the tile-value level since each tile is uniform.
    # y[i] = sum_j( sqrt(a^2 + b^2 + c^2 + d^2) ) over M_TILES
    # z[j] = sum_i( sqrt(a^2 - b^2 - c^2 - d^2) ) over N_TILES
    a2 = a_vals**2
    b2 = b_vals**2
    c2 = c_vals**2
    d2 = d_val**2

    # y_vals[i] = sum over j of sqrt(a2[i,j] + b2[i] + c2[j] + d2)
    y_golden_vals = torch.tensor(
        [
            sum(math.sqrt(float(a2[i, j] + b2[i] + c2[j] + d2)) for j in range(M_TILES))
            for i in range(N_TILES)
        ],
        dtype=torch.float32,
    )

    # z_vals[j] = sum over i of sqrt(a2[i,j] - b2[i] - c2[j] - d2)
    z_golden_vals = torch.tensor(
        [
            sum(math.sqrt(float(a2[i, j] - b2[i] - c2[j] - d2)) for i in range(N_TILES))
            for j in range(M_TILES)
        ],
        dtype=torch.float32,
    )

    # Each tile in y_result should be uniform with value y_golden_vals[tile_row]
    for i in range(N_TILES):
        tile = y_result[i * TILE_SIZE : (i + 1) * TILE_SIZE, :]
        assert torch.allclose(
            tile,
            torch.full_like(tile, y_golden_vals[i].item()),
            atol=1e-3,
        ), f"y mismatch at tile row {i}"

    # Each tile in z_result should be uniform with value z_golden_vals[tile_col]
    for j in range(M_TILES):
        tile = z_result[:, j * TILE_SIZE : (j + 1) * TILE_SIZE]
        assert torch.allclose(
            tile,
            torch.full_like(tile, z_golden_vals[j].item()),
            atol=1e-3,
        ), f"z mismatch at tile col {j}"

    print("PASSED!")


if __name__ == "__main__":
    main()
