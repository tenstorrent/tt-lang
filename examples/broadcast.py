# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Standalone example: real `import ttl`, ttnn device (not python/sim).

import torch
import ttnn


def from_torch(tensor: torch.Tensor):
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


@ttl.operation(grid="full")
def __demo_kernel(a, b, c, y):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY
    row_blocks = a.shape[0] // TILE_SIZE // row_tiles_per_block
    col_blocks = a.shape[1] // TILE_SIZE // col_tiles_per_block
    grid_x, grid_y = ttl.grid_size(dims=2)
    # divceil so a grid that does not evenly divide the blocks (e.g. an 8x7
    # compute grid, where fabric reserves a core row) still covers every block;
    # the bounds checks below drop the surplus nodes.
    row_blocks_per_node = -(-row_blocks // grid_x)
    col_blocks_per_node = -(-col_blocks // grid_y)
    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(row_tiles_per_block, col_tiles_per_block), block_count=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(row_tiles_per_block, col_tiles_per_block), block_count=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(
        c, shape=(row_tiles_per_block, 1), block_count=2
    )
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(row_tiles_per_block, col_tiles_per_block), block_count=2
    )

    @ttl.compute()
    def demo_compute():
        node_x, node_y = ttl.node(dims=2)
        for local_row_block in range(row_blocks_per_node):
            row_block = node_x * row_blocks_per_node + local_row_block
            if row_block < row_blocks:
                for local_col_block in range(col_blocks_per_node):
                    col_block = node_y * col_blocks_per_node + local_col_block
                    if col_block < col_blocks:
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            c_dfb.wait() as c_blk,
                            y_dfb.reserve() as y_blk,
                        ):
                            y_blk.store(
                                a_blk * b_blk
                                + ttl.block.broadcast(
                                    c_blk, dims=[-1], shape=y_blk.shape
                                )
                            )

    @ttl.datamovement()
    def demo_read():
        node_x, node_y = ttl.node(dims=2)
        for local_row_block in range(row_blocks_per_node):
            row_block = node_x * row_blocks_per_node + local_row_block
            if row_block < row_blocks:
                start_row_tile = row_block * row_tiles_per_block
                end_row_tile = (row_block + 1) * row_tiles_per_block
                for local_col_block in range(col_blocks_per_node):
                    col_block = node_y * col_blocks_per_node + local_col_block
                    if col_block < col_blocks:
                        start_col_tile = col_block * col_tiles_per_block
                        end_col_tile = (col_block + 1) * col_tiles_per_block
                        with (
                            a_dfb.reserve() as a_blk,
                            b_dfb.reserve() as b_blk,
                            c_dfb.reserve() as c_blk,
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
                                    0,
                                ],
                                c_blk,
                            )
                            tx_a.wait()
                            tx_b.wait()
                            tx_c.wait()

    @ttl.datamovement()
    def demo_write():
        node_x, node_y = ttl.node(dims=2)
        for local_row_block in range(row_blocks_per_node):
            row_block = node_x * row_blocks_per_node + local_row_block
            if row_block < row_blocks:
                start_row_tile = row_block * row_tiles_per_block
                end_row_tile = (row_block + 1) * row_tiles_per_block
                for local_col_block in range(col_blocks_per_node):
                    col_block = node_y * col_blocks_per_node + local_col_block
                    if col_block < col_blocks:
                        start_col_tile = col_block * col_tiles_per_block
                        end_col_tile = (col_block + 1) * col_tiles_per_block
                        with y_dfb.wait() as y_blk:
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
    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)
    c = torch.rand((shape[0], 1), dtype=torch.bfloat16)
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
