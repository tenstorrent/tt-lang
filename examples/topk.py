# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Row-wise TopK: the k largest values of each row and their column indices.

``ttl.math.topk`` mirrors ``torch.topk`` with one addition: the kernel reads
an identity index tensor (``indices[r, c] == c`` as uint16) alongside the
scores, because the device selects indices by sorting them together with the
values. The result is whole tiles, so the host slices the first ``k`` columns.
"""
import torch
import ttl
import ttnn

TILE = 32
ROWS = 32
WIDTH_TILES = 8
K = 16
LARGEST = True
OUTPUT_TILES = (K + TILE - 1) // TILE


@ttl.operation(grid=(1, 1))
def topk_rows(scores, indices, out_values, out_indices):
    scores_dfb = ttl.make_dataflow_buffer_like(
        scores, shape=(1, WIDTH_TILES), block_count=1
    )
    indices_dfb = ttl.make_dataflow_buffer_like(
        indices, shape=(1, WIDTH_TILES), block_count=1
    )
    values_out_dfb = ttl.make_dataflow_buffer_like(
        out_values, shape=(1, OUTPUT_TILES), block_count=1
    )
    indices_out_dfb = ttl.make_dataflow_buffer_like(
        out_indices, shape=(1, OUTPUT_TILES), block_count=1
    )

    @ttl.compute()
    def select():
        with (
            scores_dfb.wait() as scores_blk,
            indices_dfb.wait() as indices_blk,
            values_out_dfb.reserve() as values_blk,
            indices_out_dfb.reserve() as indices_out_blk,
        ):
            top_values, top_indices = ttl.math.topk(
                scores_blk, K, indices=indices_blk, largest=LARGEST, stable=True
            )
            values_blk.store(top_values)
            indices_out_blk.store(top_indices)

    @ttl.datamovement()
    def read():
        with scores_dfb.reserve() as scores_blk:
            ttl.copy(scores[0:1, 0:WIDTH_TILES], scores_blk).wait()
        with indices_dfb.reserve() as indices_blk:
            ttl.copy(indices[0:1, 0:WIDTH_TILES], indices_blk).wait()

    @ttl.datamovement()
    def write():
        with values_out_dfb.wait() as values_blk:
            ttl.copy(values_blk, out_values[0:1, 0:OUTPUT_TILES]).wait()
        with indices_out_dfb.wait() as indices_blk:
            ttl.copy(indices_blk, out_indices[0:1, 0:OUTPUT_TILES]).wait()


def identity_indices(rows: int, width: int) -> torch.Tensor:
    """The column number of every element, as the kernel expects."""
    columns = torch.arange(width, dtype=torch.int64)
    return columns.unsqueeze(0).expand(rows, width).to(torch.uint16).contiguous()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        width = WIDTH_TILES * TILE
        # Distinct bf16-exact values per row make the expected indices unique.
        base = torch.arange(-(width // 2), width // 2, dtype=torch.float32)
        scores_torch = torch.stack(
            [base[torch.randperm(width)] for _ in range(ROWS)]
        ).to(torch.bfloat16)

        def to_device(tensor, dtype):
            return ttnn.from_torch(
                tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
            )

        scores = to_device(scores_torch, ttnn.bfloat16)
        indices = to_device(identity_indices(ROWS, width), ttnn.uint16)
        out_width = OUTPUT_TILES * TILE
        out_values = to_device(
            torch.zeros((ROWS, out_width), dtype=torch.bfloat16), ttnn.bfloat16
        )
        out_indices = to_device(
            torch.zeros((ROWS, out_width), dtype=torch.uint16), ttnn.uint16
        )

        topk_rows(scores, indices, out_values, out_indices)

        got_values = ttnn.to_torch(out_values)[:, :K].to(torch.float32)
        got_indices = ttnn.to_torch(out_indices)[:, :K].to(torch.int64)
        expected_values, expected_indices = torch.topk(
            scores_torch.float(), K, dim=-1, largest=LARGEST, sorted=True
        )
        assert torch.equal(got_values, expected_values), "value mismatch"
        assert torch.equal(got_indices, expected_indices), "index mismatch"
        print("PASSED!")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
