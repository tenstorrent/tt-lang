# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# type: ignore
import torch
import math

from sim import ttl


@ttl.kernel(
    grid="auto",  # Let compiler choose grid
)
def tt_lang_multicore_matmul(
    a_in: torch.Tensor,
    b_in: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # Validate shapes at a high level.
    assert a_in.ndim == 2 and b_in.ndim == 2 and out.ndim == 2
    assert (
        a_in.shape[1] == b_in.shape[0]
    ), "Incompatible matrix shapes for multiplication."
    assert (a_in.shape[0], b_in.shape[1]) == out.shape, "Output shape must be (M, N)."
    # Validate tiling.
    assert all(ttl.is_tiled(t, ttl.TILE_SHAPE) for t in [a_in, b_in, out])

    # Tile counts.
    Mt = a_in.shape[0] // ttl.TILE_SHAPE[0]
    Kt = a_in.shape[1] // ttl.TILE_SHAPE[1]
    Nt = b_in.shape[1] // ttl.TILE_SHAPE[1]

    # Accessors in tile index space.
    a_accessor = ttl.TensorAccessor(a_in, index_type=ttl.IndexType.TILE)
    b_accessor = ttl.TensorAccessor(b_in, index_type=ttl.IndexType.TILE)
    out_accessor = ttl.TensorAccessor(out, index_type=ttl.IndexType.TILE)

    # Work splitting logic
    grid_h, grid_w = ttl.grid_size()
    total_cores = grid_h * grid_w
    total_output_tiles = Mt * Nt

    # This logic replicates ttnn.split_work_to_cores
    num_tiles_per_core_base = total_output_tiles // total_cores
    remainder_tiles = total_output_tiles % total_cores

    # Each core needs to know its start tile and how many tiles to process
    core_work = []
    start_tile_id = 0
    for i in range(total_cores):
        num_tiles_this_core = num_tiles_per_core_base + (
            1 if i < remainder_tiles else 0
        )
        core_work.append((start_tile_id, num_tiles_this_core))
        start_tile_id += num_tiles_this_core

    # Circular buffers (single-tile, double-buffered).
    buffering_factor = 2
    a_cb = ttl.make_circular_buffer_like(
        a_in, shape=(1, 1), buffer_factor=buffering_factor
    )
    b_cb = ttl.make_circular_buffer_like(
        b_in, shape=(1, 1), buffer_factor=buffering_factor
    )
    out_cb = ttl.make_circular_buffer_like(
        out, shape=(1, 1), buffer_factor=buffering_factor
    )

    @ttl.compute()
    def mm_compute():
        core_id = ttl.core(dims=1)  # Linear core index
        _, num_tiles_for_core = core_work[core_id]

        for _ in range(num_tiles_for_core):
            out_block = out_cb.reserve()  # blocking
            # Accumulate K contributions for one output tile
            for k in range(Kt):
                a_block = a_cb.wait()  # blocking
                b_block = b_cb.wait()  # blocking

                if k == 0:
                    # Initialize output tile
                    out_block.store(a_block @ b_block)
                else:
                    # Accumulate into the output tile
                    out_block.store(out_block + (a_block @ b_block))

                a_cb.pop()
                b_cb.pop()
            out_cb.push()

    @ttl.datamovement()
    def mm_reader():
        core_id = ttl.core(dims=1)
        start_tile_id, num_tiles_for_core = core_work[core_id]

        for tile_offset in range(num_tiles_for_core):
            output_tile_id = start_tile_id + tile_offset
            out_row = output_tile_id // Nt
            out_col = output_tile_id % Nt

            for k in range(Kt):
                a_blk = a_cb.reserve()
                b_blk = b_cb.reserve()

                a_wr = ttl.copy(
                    a_accessor[slice(out_row, out_row + 1), slice(k, k + 1)], a_blk
                )
                b_wr = ttl.copy(
                    b_accessor[slice(k, k + 1), slice(out_col, out_col + 1)], b_blk
                )

                a_wr.wait()
                b_wr.wait()

                a_cb.push()
                b_cb.push()

    @ttl.datamovement()
    def mm_writer():
        core_id = ttl.core(dims=1)
        start_tile_id, num_tiles_for_core = core_work[core_id]

        for tile_offset in range(num_tiles_for_core):
            output_tile_id = start_tile_id + tile_offset
            out_row = output_tile_id // Nt
            out_col = output_tile_id % Nt

            out_blk = out_cb.wait()
            out_wr = ttl.copy(
                out_blk,
                out_accessor[slice(out_row, out_row + 1), slice(out_col, out_col + 1)],
            )
            out_wr.wait()
            out_cb.pop()

    # Execute the program on the grid.
    ttl.Program(mm_compute, mm_reader, mm_writer)(a_in, b_in, out)


if __name__ == "__main__":
    from sim.testing import assert_pcc

    # Use parameters that are tile-divisible
    dim_m = 128
    dim_k = 256
    dim_n = 64

    a_in = torch.randn(dim_m, dim_k)
    b_in = torch.randn(dim_k, dim_n)
    out = torch.zeros(dim_m, dim_n)

    tt_lang_multicore_matmul(a_in, b_in, out)

    golden = torch.matmul(a_in, b_in)
    assert_pcc(golden, out, rtol=1e-4, atol=1e-4)
    print("Multi-core matmul test passed!")
