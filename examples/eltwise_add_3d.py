# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2
BATCH_GRANULARITY = 2  # Number of batch elements per block


@ttl.operation(grid="auto")
def eltwise_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    """Element-wise addition TT-Lang operation for 3D tensors (batch, rows, cols).

    Processes tensors with shape (batch_size, height, width) using 3D blocks.
    The batch, row, and column dimensions are all processed in blocks, with
    rows and cols distributed across nodes using a 2D grid.
    """
    batch_tiles = a_in.shape[0] // BATCH_GRANULARITY
    row_tiles = a_in.shape[1] // TILE_SIZE // GRANULARITY
    col_tiles = a_in.shape[2] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(
        a_in, shape=(BATCH_GRANULARITY, GRANULARITY, 1), buffer_factor=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b_in, shape=(BATCH_GRANULARITY, GRANULARITY, 1), buffer_factor=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(BATCH_GRANULARITY, GRANULARITY, 1), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for batch in range(batch_tiles):
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            with (
                                a_dfb.wait() as a_blk,
                                b_dfb.wait() as b_blk,
                                out_dfb.reserve() as out_blk,
                            ):
                                out_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for batch in range(batch_tiles):
            b0, b1 = batch * BATCH_GRANULARITY, (batch + 1) * BATCH_GRANULARITY
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                                tx_a = ttl.copy(
                                    a_in[b0:b1, r0:r1, col : col + 1], a_blk
                                )
                                tx_b = ttl.copy(
                                    b_in[b0:b1, r0:r1, col : col + 1], b_blk
                                )
                                tx_a.wait()
                                tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for batch in range(batch_tiles):
            b0, b1 = batch * BATCH_GRANULARITY, (batch + 1) * BATCH_GRANULARITY
            for local_row in range(rows_per_node):
                row = node_row * rows_per_node + local_row
                if row < row_tiles:
                    r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                    for local_col in range(cols_per_node):
                        col = node_col * cols_per_node + local_col
                        if col < col_tiles:
                            with out_dfb.wait() as out_blk:
                                tx = ttl.copy(out_blk, out[b0:b1, r0:r1, col : col + 1])
                                tx.wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        batch = 8  # Must be a multiple of BATCH_GRANULARITY (2)
        dim = 256  # Height and width must be tile-aligned (multiples of 32)
        a_torch = torch.rand((batch, dim, dim), dtype=torch.bfloat16)
        b_torch = torch.rand((batch, dim, dim), dtype=torch.bfloat16)

        a = ttnn.from_torch(
            a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        b = ttnn.from_torch(
            b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out = ttnn.from_torch(
            torch.zeros_like(a_torch),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        eltwise_add(a, b, out)

        result = ttnn.to_torch(out)
        expected = a_torch + b_torch

        assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2), "Mismatch!"
        print("PASSED!")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
