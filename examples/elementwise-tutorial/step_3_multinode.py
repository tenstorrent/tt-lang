# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

#
# Tutorial Step 3: Multi-Node, Fixed Grid
# ========================================
# Extends Step 2 by running the operation across a grid of nodes in parallel.
#
# New concepts introduced:
#   - grid=(4, 4)          — run the operation on a 4×4 grid of nodes (16 cores)
#   - ttl.grid_size(dims=2) — query the grid dimensions at runtime
#   - ttl.node(dims=2)     — query this node's (col, row) position in the grid
#
# Each node processes an independent rectangular region of the output tensor.
# The partition is computed from the node's coordinates so that all nodes cover
# the full tensor without overlap.  This requires the tensor dimensions to be
# evenly divisible by the grid (see Step 4 for a version that handles remainders).

import ttnn
import torch


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


# grid=(4, 4) launches the operation body on every node of a 4-column × 4-row grid.
# All nodes execute the same code; they differentiate their work via ttl.node().


@ttl.operation(grid=(4, 4))
def __tutorial_operation(
    a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor
):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    # ttl.grid_size returns the number of nodes along each grid dimension.
    # dims=2 returns (cols, rows) matching the (col, row) convention of ttl.node.

    grid_cols, grid_rows = ttl.grid_size(dims=2)

    # Divide the total block count evenly across the grid.
    # Assumes the tensor is evenly divisible by the grid size.

    rows_per_node = a.shape[0] // TILE_SIZE // row_tiles_per_block // grid_rows
    cols_per_node = a.shape[1] // TILE_SIZE // col_tiles_per_block // grid_rows

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

    # The compute kernel is unaware of node coordinates; it simply processes
    # all blocks assigned to this node by the DM threads.

    @ttl.compute()
    def tutorial_compute():
        for _ in range(rows_per_node):
            for _ in range(cols_per_node):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    c_dfb.wait() as c_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    y_blk.store(a_blk * b_blk + c_blk)

    @ttl.datamovement()
    def tutorial_read():

        # ttl.node() returns the zero-based coordinates of this specific node.
        # node_col and node_row are used to offset into the global tensor.

        node_col, node_row = ttl.node(dims=2)

        for local_row in range(rows_per_node):

            # Map local block index to global block index.

            row = node_row * rows_per_node + local_row
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for local_col in range(cols_per_node):
                col = node_col * cols_per_node + local_col
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
            start_row_tile = row * row_tiles_per_block
            end_row_tile = (row + 1) * row_tiles_per_block

            for local_col in range(cols_per_node):
                col = node_col * cols_per_node + local_col
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


def tutorial_operation(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor):
    y = from_torch(torch.zeros((a.shape[0], a.shape[1]), dtype=torch.bfloat16))
    __tutorial_operation(a, b, c, y)
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

    y = ttnn.multiply(tutorial_operation(a, b, c), d)

    y = ttnn.to_torch(y)
    print(y)
    print(expected_y)

    assert torch.allclose(y, expected_y, rtol=1e-2, atol=1e-2), "Tensors do not match"

finally:
    ttnn.close_device(device)
