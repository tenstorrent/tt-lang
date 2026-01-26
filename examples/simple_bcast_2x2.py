# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Broadcast test: 2x2 tiles for full tensor, 2x1 tiles for broadcast input.

Reader loads both broadcast tiles once, then reuses them for all output tiles.
"""
import ttnn
import torch

import ttl

TILE_SIZE = 32
ROW_TILES_PER_BLOCK = 2
COL_TILES_PER_BLOCK = 2


@ttl.kernel(grid=(1, 1))
def simple_bcast_2x2_kernel(a, b, y):
    row_tiles_per_block = ROW_TILES_PER_BLOCK
    col_tiles_per_block = COL_TILES_PER_BLOCK

    row_tiles = a.shape[0] // TILE_SIZE
    col_tiles = a.shape[1] // TILE_SIZE

    rows_per_core = row_tiles // row_tiles_per_block
    cols_per_core = col_tiles // col_tiles_per_block

    a_cb = ttl.make_circular_buffer_like(
        a, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )
    b_cb = ttl.make_circular_buffer_like(
        b, shape=(row_tiles_per_block, 1), buffer_factor=2
    )
    y_cb = ttl.make_circular_buffer_like(
        y, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        with b_cb.wait() as b_blk:
            for _ in range(rows_per_core):
                for _ in range(cols_per_core):
                    with (
                        a_cb.wait() as a_blk,
                        y_cb.reserve() as y_blk,
                    ):
                        y_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def reader():
        with b_cb.reserve() as b_blk:
            tx_b = ttl.copy(b[0:2, 0:1], b_blk)
            tx_b.wait()

        for local_row in range(rows_per_core):
            start_row_tile = local_row * row_tiles_per_block
            end_row_tile = (local_row + 1) * row_tiles_per_block

            for local_col in range(cols_per_core):
                start_col_tile = local_col * col_tiles_per_block
                end_col_tile = (local_col + 1) * col_tiles_per_block

                with a_cb.reserve() as a_blk:
                    tx_a = ttl.copy(
                        a[start_row_tile:end_row_tile, start_col_tile:end_col_tile],
                        a_blk,
                    )
                    tx_a.wait()

    @ttl.datamovement()
    def writer():
        for local_row in range(rows_per_core):
            start_row_tile = local_row * row_tiles_per_block
            end_row_tile = (local_row + 1) * row_tiles_per_block

            for local_col in range(cols_per_core):
                start_col_tile = local_col * col_tiles_per_block
                end_col_tile = (local_col + 1) * col_tiles_per_block

                with y_cb.wait() as y_blk:
                    tx = ttl.copy(
                        y_blk,
                        y[start_row_tile:end_row_tile, start_col_tile:end_col_tile],
                    )
                    tx.wait()


def run_simple_bcast_2x2(a, b, device):
    y = ttnn.from_torch(
        torch.zeros((a.shape[0], a.shape[1]), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    simple_bcast_2x2_kernel(a, b, y)
    return y


def main() -> None:
    device = ttnn.open_device(device_id=0)

    try:
        tensor_rows = TILE_SIZE * 2
        tensor_cols = TILE_SIZE * 2

        a_torch = (
            torch.arange(tensor_cols, dtype=torch.float32)
            .unsqueeze(0)
            .expand(tensor_rows, tensor_cols)
            % TILE_SIZE
        ).to(torch.bfloat16)

        b_torch = torch.zeros((tensor_rows, TILE_SIZE), dtype=torch.bfloat16)
        b_torch[:, 0] = torch.arange(tensor_rows, dtype=torch.float32).to(
            torch.bfloat16
        )

        expected = a_torch + b_torch[:, :1]

        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        y = run_simple_bcast_2x2(a, b, device)

        y_torch = ttnn.to_torch(y)

        if torch.allclose(y_torch, expected, rtol=1e-2, atol=1e-2):
            print("PASS: Simple broadcast 2x2 test passed!")
        else:
            diff = torch.abs(y_torch - expected)
            max_diff_idx = torch.argmax(diff)
            row = max_diff_idx // tensor_cols
            col = max_diff_idx % tensor_cols
            print(f"FAIL: Mismatch at ({row}, {col})")
            print(f"  Expected: {expected[row, col]}, Got: {y_torch[row, col]}")
            print(f"  Max diff: {diff.max()}")
            raise AssertionError("Simple broadcast 2x2 test failed")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
