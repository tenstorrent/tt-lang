# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch


def from_torch(tensor: ttnn.Tensor):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


import ttl

TILE_SIZE = 32
GRANULARITY = 4


@ttl.kernel(grid=(4, 4))
def __demo_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    grid_cols, grid_rows = ttl.grid_size(dims=2)

    rows_per_core = a.shape[0] // TILE_SIZE // row_tiles_per_block // grid_rows
    cols_per_core = a.shape[1] // TILE_SIZE // col_tiles_per_block // grid_rows

    a_cb = ttl.make_circular_buffer_like(
        a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    c_cb = ttl.make_circular_buffer_like(
        c, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    y_cb = ttl.make_circular_buffer_like(
        y, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )

    @ttl.compute()
    def demo_compute():
        for _ in range(rows_per_core):
            for _ in range(cols_per_core):
                with (
                    a_cb.wait() as a_blk,
                    b_cb.wait() as b_blk,
                    c_cb.wait() as c_blk,
                    y_cb.reserve() as y_blk,
                ):
                    y_blk.store(a_blk * b_blk + c_blk)

    @ttl.datamovement()
    def demo_read():
        core_col, core_row = ttl.core(dims=2)

        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for local_col in range(cols_per_core):
                col = core_col * cols_per_core + local_col
                start_col_tile = col * col_tiles_per_block
                end_col_tile = (col + 1) * col_tiles_per_block

                with (
                    a_cb.reserve() as a_blk,
                    b_cb.reserve() as b_blk,
                    c_cb.reserve() as c_blk,
                ):
                    tx_a = ttl.copy(
                        a[
                            start_row_tile:end_row_tile,
                            start_col_tile:end_col_tile,
                        ],
                        a_blk,
                    )
                    tx_b = ttl.copy(
                        b[
                            start_row_tile:end_row_tile,
                            start_col_tile:end_col_tile,
                        ],
                        b_blk,
                    )
                    tx_c = ttl.copy(
                        c[
                            start_row_tile:end_row_tile,
                            start_col_tile:end_col_tile,
                        ],
                        c_blk,
                    )

                    tx_a.wait()
                    tx_b.wait()
                    tx_c.wait()

    @ttl.datamovement()
    def demo_write():
        core_col, core_row = ttl.core(dims=2)

        for local_row in range(rows_per_core):
            row = core_row * rows_per_core + local_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for local_col in range(cols_per_core):
                col = core_col * cols_per_core + local_col
                start_col_tile = col * col_tiles_per_block
                end_col_tile = (col + 1) * col_tiles_per_block

                with y_cb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk,
                        y[
                            start_row_tile:end_row_tile,
                            start_col_tile:end_col_tile,
                        ],
                    )
                    tx.wait()


def demo_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor):
    y = from_torch(torch.zeros((a.shape[0], a.shape[1]), dtype=torch.bfloat16))
    __demo_kernel(a, b, c, y)
    return y


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)

try:
    shape = (2048, 2048)

    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)
    c = torch.rand(shape, dtype=torch.bfloat16)
    d = torch.rand(shape, dtype=torch.bfloat16)

    expected_y = (a * b + c) * d

    a = from_torch(a)
    b = from_torch(b)
    c = from_torch(c)
    d = from_torch(d)

    y = ttnn.multiply(demo_kernel(a, b, c), d)

    y = ttnn.to_torch(y)
    print(y)
    print(expected_y)

    assert torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2), "Tensors do not match"

finally:
    ttnn.close_device(device)
