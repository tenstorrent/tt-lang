# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from sim import ttl


@ttl.kernel(
    grid=(1, 1),  # single-core
)
def tt_lang_singlecore_matmul(
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
    Kt_a = a_in.shape[1] // ttl.TILE_SHAPE[1]
    Kt_b = b_in.shape[0] // ttl.TILE_SHAPE[0]
    Nt = b_in.shape[1] // ttl.TILE_SHAPE[1]
    assert Kt_a == Kt_b, "K tile counts across A (cols) and B (rows) must match."
    Kt = Kt_a

    # Accessors in tile index space.
    a_accessor = ttl.TensorAccessor(a_in, index_type=ttl.IndexType.TILE)
    b_accessor = ttl.TensorAccessor(b_in, index_type=ttl.IndexType.TILE)
    out_accessor = ttl.TensorAccessor(out, index_type=ttl.IndexType.TILE)

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
        # Compute each output tile (m, n).
        for _ in range(Mt):
            for _ in range(Nt):
                # Reserve output tile once and accumulate K contributions.
                out_block = out_cb.reserve()  # blocking
                for k in range(Kt):
                    a_block = a_cb.wait()  # blocking
                    b_block = b_cb.wait()  # blocking

                    if k == 0:
                        # Initialize output tile.
                        out_block.store(a_block @ b_block)
                    else:
                        # Accumulate into the output tile.
                        out_block.store(out_block + (a_block @ b_block))

                    # Free input tiles.
                    a_cb.pop()
                    b_cb.pop()

                # Finalize the output tile.
                out_cb.push()

    @ttl.datamovement()
    def mm_reader():
        # Feed input tiles for each (m, n, k) triple.
        for m in range(Mt):
            for n in range(Nt):
                for k in range(Kt):
                    a_blk = a_cb.reserve()
                    b_blk = b_cb.reserve()
                    # Tile indices in tile space.
                    a_wr = ttl.copy(a_accessor[slice(m, m + 1), slice(k, k + 1)], a_blk)
                    b_wr = ttl.copy(b_accessor[slice(k, k + 1), slice(n, n + 1)], b_blk)
                    a_wr.wait()
                    b_wr.wait()
                    a_cb.push()
                    b_cb.push()

    @ttl.datamovement()
    def mm_writer():
        # Write each completed output tile.
        for m in range(Mt):
            for n in range(Nt):
                out_blk = out_cb.wait()
                out_wr = ttl.copy(
                    out_blk, out_accessor[slice(m, m + 1), slice(n, n + 1)]
                )
                out_wr.wait()
                out_cb.pop()

    # Execute the program on the single core.
    ttl.Program(mm_compute, mm_reader, mm_writer)(a_in, b_in, out)


if __name__ == "__main__":
    from sim.testing import assert_pcc

    # Use parameters that match the singlecore_matmul requirements
    dim_m = 128
    dim_k = 256
    dim_n = 64
    a_in = torch.randn(dim_m, dim_k)
    b_in = torch.randn(dim_k, dim_n)
    out = torch.zeros(dim_m, dim_n)

    tt_lang_singlecore_matmul(a_in, b_in, out)

    golden = torch.matmul(a_in, b_in)
    assert_pcc(golden, out, rtol=1e-4, atol=1e-4)
    print("Single-core matmul test passed!")
