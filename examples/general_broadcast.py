# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch


def from_torch(t):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


import ttl

TILE_SIZE = 32
GRANULARITY = 4


@ttl.kernel(grid=(8, 8))
def __demo_kernel(a, b, c, y):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY
    grid_x, grid_y = ttl.grid_size(dims=2)
    rows_per_core = a.shape[0] // TILE_SIZE // grid_x // row_tiles_per_block
    cols_per_core = a.shape[1] // TILE_SIZE // grid_y // col_tiles_per_block

    # Determine CB shapes based on tensor dimensions
    b_row_tiles = row_tiles_per_block if b.shape[0] > TILE_SIZE else 1
    b_col_tiles = col_tiles_per_block if b.shape[1] > TILE_SIZE else 1
    c_row_tiles = row_tiles_per_block if c.shape[0] > TILE_SIZE else 1
    c_col_tiles = col_tiles_per_block if c.shape[1] > TILE_SIZE else 1

    a_cb = ttl.make_circular_buffer_like(
        a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(b_row_tiles, b_col_tiles), buffer_factor=2
    )
    c_cb = ttl.make_circular_buffer_like(
        c, shape=(c_row_tiles, c_col_tiles), buffer_factor=2
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
                    # Handle broadcasting based on actual shapes
                    # If b or c have size 1 in any dimension, broadcast them
                    b_expr = b_blk
                    c_expr = c_blk

                    # Check if b needs broadcasting
                    if b_row_tiles == 1 and row_tiles_per_block > 1:
                        b_expr = ttl.math.broadcast(b_blk, dims=[0])
                    if b_col_tiles == 1 and col_tiles_per_block > 1:
                        b_expr = ttl.math.broadcast(
                            b_expr if b_expr != b_blk else b_blk,
                            dims=[1] if b_row_tiles > 1 else [0, 1],
                        )

                    # Check if c needs broadcasting
                    if c_row_tiles == 1 and row_tiles_per_block > 1:
                        c_expr = ttl.math.broadcast(c_blk, dims=[0])
                    if c_col_tiles == 1 and col_tiles_per_block > 1:
                        c_expr = ttl.math.broadcast(
                            c_expr if c_expr != c_blk else c_blk,
                            dims=[1] if c_row_tiles > 1 else [0, 1],
                        )

                    y_blk.store(a_blk * b_expr + c_expr)

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
                    # Handle broadcasting for b
                    b_start_row = 0 if b.shape[0] <= TILE_SIZE else start_row_tile
                    b_end_row = b_row_tiles if b.shape[0] <= TILE_SIZE else end_row_tile
                    b_start_col = 0 if b.shape[1] <= TILE_SIZE else start_col_tile
                    b_end_col = b_col_tiles if b.shape[1] <= TILE_SIZE else end_col_tile
                    tx_b = ttl.copy(
                        b[
                            b_start_row:b_end_row,
                            b_start_col:b_end_col,
                        ],
                        b_blk,
                    )
                    # Handle broadcasting for c
                    c_start_row = 0 if c.shape[0] <= TILE_SIZE else start_row_tile
                    c_end_row = c_row_tiles if c.shape[0] <= TILE_SIZE else end_row_tile
                    c_start_col = 0 if c.shape[1] <= TILE_SIZE else start_col_tile
                    c_end_col = c_col_tiles if c.shape[1] <= TILE_SIZE else end_col_tile
                    tx_c = ttl.copy(
                        c[
                            c_start_row:c_end_row,
                            c_start_col:c_end_col,
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


def demo_kernel(a, b, c):
    y = from_torch(torch.zeros((a.shape[0], a.shape[1]), dtype=torch.bfloat16))
    __demo_kernel(a, b, c, y)
    return y


torch.manual_seed(42)
device = ttnn.open_device(device_id=0)
try:
    shape = (2048, 2048)
    # Test 1: Broadcasting c with shape (shape[0], 1)
    print("Test 1: Broadcasting c with shape (2048, 1)")
    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)
    c = torch.rand((shape[0], 1), dtype=torch.bfloat16)
    d = torch.rand(shape, dtype=torch.bfloat16)
    expected_y = (a * b + c) * d
    a_tt = from_torch(a)
    b_tt = from_torch(b)
    c_tt = from_torch(c)
    d_tt = from_torch(d)
    y = ttnn.multiply(demo_kernel(a_tt, b_tt, c_tt), d_tt)
    y = ttnn.to_torch(y)
    print(
        "Test 1 passed!"
        if torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2)
        else "Test 1 FAILED!"
    )
    assert torch.allclose(
        y, expected_y, rtol=1e-2, atol=1e-2
    ), "Test 1: Tensors do not match"

    # Test 2: Broadcasting b with shape (1, shape[1])
    print("\nTest 2: Broadcasting b with shape (1, 2048)")
    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand((1, shape[1]), dtype=torch.bfloat16)
    c = torch.rand((shape[0], 1), dtype=torch.bfloat16)
    d = torch.rand(shape, dtype=torch.bfloat16)
    expected_y = (a * b + c) * d
    a_tt = from_torch(a)
    b_tt = from_torch(b)
    c_tt = from_torch(c)
    d_tt = from_torch(d)
    y = ttnn.multiply(demo_kernel(a_tt, b_tt, c_tt), d_tt)
    y = ttnn.to_torch(y)
    print(
        "Test 2 passed!"
        if torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2)
        else "Test 2 FAILED!"
    )
    assert torch.allclose(
        y, expected_y, rtol=1e-2, atol=1e-2
    ), "Test 2: Tensors do not match"
finally:
    ttnn.close_device(device)
