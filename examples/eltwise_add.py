# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@ttl.kernel(grid="auto")
def eltwise_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = a_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_core = -(-row_tiles // grid_rows)
    cols_per_core = -(-col_tiles // grid_cols)

    a_cb = ttl.make_circular_buffer_like(a_in, shape=(GRANULARITY, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b_in, shape=(GRANULARITY, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(GRANULARITY, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        core_col, core_row = ttl.core(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < row_tiles:
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < col_tiles:
                        with a_cb.wait() as a_blk, b_cb.wait() as b_blk, out_cb.reserve() as out_blk:
                            out_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def read():
        core_col, core_row = ttl.core(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < col_tiles:
                        with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                            tx_a = ttl.copy(a_in[r0:r1, col : col + 1], a_blk)
                            tx_b = ttl.copy(b_in[r0:r1, col : col + 1], b_blk)
                            tx_a.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def write():
        core_col, core_row = ttl.core(dims=2)
        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_core):
                    col = core_col * cols_per_core + local_col
                    if col < col_tiles:
                        with out_cb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        dim = 256
        a_torch = torch.rand((dim, dim), dtype=torch.bfloat16)
        b_torch = torch.rand((dim, dim), dtype=torch.bfloat16)

        a = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(torch.zeros_like(a_torch), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        eltwise_add(a, b, out)

        result = ttnn.to_torch(out)
        expected = a_torch + b_torch
        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), "Mismatch!"
        print("PASSED!")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
