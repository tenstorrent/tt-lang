# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# type: ignore

import ttnn
import torch

import ttl
from ttl import copy, Program, make_circular_buffer_like


@ttl.kernel(grid=(1, 1))
def add_with_kernel(a, b, y):
    row_tiles = 2
    col_tiles = 2

    rows = 2
    cols = 2

    a_cb = make_circular_buffer_like(a, shape=(row_tiles, col_tiles), buffer_factor=2)
    b_cb = make_circular_buffer_like(b, shape=(row_tiles, col_tiles), buffer_factor=2)
    y_cb = make_circular_buffer_like(y, shape=(row_tiles, col_tiles), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        for _ in range(rows):
            for _ in range(cols):
                with a_cb.wait() as a, b_cb.wait() as b, y_cb.reserve() as y:
                    y.store(a + b)

    @ttl.datamovement()
    def add_read():
        for r in range(rows):
            for c in range(cols):
                with a_cb.reserve() as a_block, b_cb.reserve() as b_block:
                    tx_a = copy(a[r, c], a_block)
                    tx_b = copy(b[r, c], b_block)

                    tx_a.wait()
                    tx_b.wait()

    @ttl.datamovement()
    def add_write():
        for r in range(rows):
            for c in range(cols):
                with y_cb.wait() as y_block:
                    tx = copy(y_cb, y[r, c])
                    tx.wait()

    return Program(add_compute, add_read, add_write)(a, b, y)


device = ttnn.open_device(device_id=0)

try:
    # shape = (64, 64)
    shape = (128, 128)
    a = torch.full(shape, 2.0, dtype=torch.bfloat16)
    b = torch.full(shape, 3.0, dtype=torch.bfloat16)
    y = torch.zeros(shape, dtype=torch.bfloat16)

    expected = a + b

    a = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    b = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y = ttnn.from_torch(
        y,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    add_with_kernel(a, b, y)

    y = ttnn.to_torch(y)
    print(y)
    print(expected)

finally:
    ttnn.close_device(device)
