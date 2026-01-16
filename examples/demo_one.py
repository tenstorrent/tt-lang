# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch

import ttl

TILE_SIZE = 32
GRANULARITY = 4


@ttl.kernel(grid=(8, 8))
def demo_kernel(a, b, c, y):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    grid_x, grid_y = ttl.grid_size(dims=2)

    rows_per_core = a.shape[0] // TILE_SIZE // grid_x // row_tiles_per_block
    cols_per_core = a.shape[1] // TILE_SIZE // grid_y // col_tiles_per_block

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
        core_x, core_y = ttl.core(dims=2)

        for core_row in range(rows_per_core):
            row = core_x * rows_per_core + core_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for core_col in range(cols_per_core):
                col = core_y * cols_per_core + core_col
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
        core_x, core_y = ttl.core(dims=2)

        for core_row in range(rows_per_core):
            row = core_x * rows_per_core + core_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for core_col in range(cols_per_core):
                col = core_y * cols_per_core + core_col
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

    return ttl.Program(demo_compute, demo_read, demo_write)(a, b, c, y)


torch.manual_seed(42)

device = ttnn.open_device(device_id=0)


def from_torch(t):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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
    t = from_torch(torch.zeros(shape, dtype=torch.bfloat16))
    y = from_torch(torch.zeros(shape, dtype=torch.bfloat16))

    demo_kernel(a, b, c, t)
    y = ttnn.multiply(t, d)

    y = ttnn.to_torch(y)
    print(y)
    print(expected_y)

    assert torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2), "Tensors do not match"

finally:
    ttnn.close_device(device)
