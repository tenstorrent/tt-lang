# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import torch

from ttlang import ttl


@ttl.kernel(grid=(1, 1))
def add_with_kernel(a, b, c, y):
    # row_tiles = 1
    # col_tiles = 1

    row_tiles = 2
    col_tiles = 2

    rows = a.shape[0] // 32 // row_tiles
    cols = a.shape[1] // 32 // col_tiles

    a_cb = ttl.make_circular_buffer_like(
        a, shape=(row_tiles, col_tiles), buffer_factor=2
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(row_tiles, col_tiles), buffer_factor=2
    )
    c_cb = ttl.make_circular_buffer_like(
        c, shape=(row_tiles, col_tiles), buffer_factor=2
    )
    y_cb = ttl.make_circular_buffer_like(
        y, shape=(row_tiles, col_tiles), buffer_factor=2
    )

    @ttl.compute()
    def add_compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    a_cb.wait() as a_block,
                    b_cb.wait() as b_block,
                    c_cb.wait() as c_block,
                    y_cb.reserve() as y,
                ):
                    y.store(a_block * b_block + c_block)

    @ttl.datamovement()
    def add_read():
        for row in range(rows):
            for col in range(cols):
                with (
                    a_cb.reserve() as a_blk,
                    b_cb.reserve() as b_blk,
                    c_cb.reserve() as c_blk,
                ):
                    tx_a = ttl.copy(
                        a[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                        a_blk,
                    )
                    tx_b = ttl.copy(
                        b[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                        b_blk,
                    )
                    tx_c = ttl.copy(
                        c[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                        c_blk,
                    )

                    tx_a.wait()
                    tx_b.wait()
                    tx_c.wait()

    @ttl.datamovement()
    def add_write():
        for row in range(rows):
            for col in range(cols):
                with y_cb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk,
                        y[
                            row * row_tiles : (row + 1) * row_tiles,
                            col * col_tiles : (col + 1) * col_tiles,
                        ],
                    )
                    tx.wait()

    return ttl.Program(add_compute, add_read, add_write)(a, b, c, y)


device = ttnn.open_device(device_id=0)

try:
    shape = (2048, 2048)
    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)
    c = torch.rand(shape, dtype=torch.bfloat16)
    y = torch.zeros(shape, dtype=torch.bfloat16)

    expected_y = a * b + c

    a = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c = ttnn.from_torch(
        c,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y = ttnn.from_torch(
        y,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    add_with_kernel(a, b, c, y)

    y = ttnn.to_torch(y)
    print(y)
    print(expected_y)

    assert torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2), "Tensors do not match"

finally:
    ttnn.close_device(device)
