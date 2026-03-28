# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 4: Multi-Node, Auto Grid
# =======================================
# Extends Step 3 by removing the hard-coded grid size and handling tensor
# dimensions that are not evenly divisible by the grid.
#
# New concepts introduced:
#   - grid="auto"     — the compiler picks the largest grid available in the;
#                       the kernel must not assume any specific grid dimensions
#   - ceiling division — ensures every block is assigned to a node even when
#                        the block count doesn't divide evenly across the grid
#   - bounds checking — nodes at the trailing edge of the grid may have fewer
#                       blocks to process; guard all per-block work with
#                       `if row < rows` / `if col < cols`
#
# Because compute and DM threads must agree on which blocks to process, the
# bounds check appears in all three thread functions.

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


# grid="auto" asks the compiler to select the grid at compile time based on
# available hardware resources.  The kernel body must work correctly for any
# grid the compiler may choose.


@ttl.kernel(grid="auto")
def __tutorial_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    grid_cols, grid_rows = ttl.grid_size(dims=2)

    # Total block counts across the entire tensor (not per-node).

    rows = a.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = a.shape[1] // TILE_SIZE // col_tiles_per_block

    # Ceiling division: -(-x // y) is a concise Python idiom for ceil(x / y).
    # This ensures every block is covered even when rows/cols is not a multiple
    # of the grid size.  Nodes in the last row/column of the grid may receive
    # fewer blocks and rely on the bounds checks below to skip out-of-range work.

    rows_per_node = -(-rows // grid_rows)
    cols_per_node = -(-cols // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(
        c, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )

    @ttl.compute()
    def tutorial_compute():
        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row

            # Skip if this node was assigned more iterations than there are
            # actual blocks (happens at the trailing edge of the grid).

            if row < rows:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < cols:
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            c_dfb.wait() as c_blk,
                            y_dfb.reserve() as y_blk,
                        ):
                            y_blk.store(a_blk * b_blk + c_blk)

    @ttl.datamovement()
    def tutorial_read():
        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < rows:
                start_row_tile = row * row_tiles_per_block
                end_row_tile = (row + 1) * row_tiles_per_block

                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < cols:
                        start_col_tile = col * col_tiles_per_block
                        end_col_tile = (col + 1) * col_tiles_per_block

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
                                    start_col_tile:end_col_tile,
                                ],
                                c_blk,
                            )

                            tx_a.wait()
                            tx_b.wait()
                            tx_c.wait()

    @ttl.datamovement()
    def tutorial_write():
        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < rows:
                start_row_tile = row * row_tiles_per_block
                end_row_tile = (row + 1) * row_tiles_per_block

                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < cols:
                        start_col_tile = col * col_tiles_per_block
                        end_col_tile = (col + 1) * col_tiles_per_block

                        with y_dfb.wait() as y_blk:
                            tx = ttl.copy(
                                y_blk,
                                y[
                                    start_row_tile:end_row_tile,
                                    start_col_tile:end_col_tile,
                                ],
                            )
                            tx.wait()


def tutorial_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor):
    y = from_torch(torch.zeros((a.shape[0], a.shape[1]), dtype=torch.bfloat16))
    __tutorial_kernel(a, b, c, y)
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

    y = ttnn.multiply(tutorial_kernel(a, b, c), d)

    y = ttnn.to_torch(y)
    print(y)
    print(expected_y)

    assert torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2), "Tensors do not match"

finally:
    ttnn.close_device(device)
