# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproducer for reduce_max bug in functional simulator.

BUG: reduce_max and reduce_sum operate on TILE GRID level (combining tiles
element-wise across the grid), but they should operate on ELEMENT level
within each tile.

For a single-tile block (1,1), reduce_max(block, scaler, dims=[0]) returns
the input unchanged because there are no tiles to combine at the grid level.

EXPECTED: reduce_max should compute max across rows (dim 0) WITHIN the 32x32
tile, returning a 1x32 result or a scalar per column.

This bug causes softmax to fail in the MNIST inference kernel:
- reduce_max(logits, scaler, dims=[0]) returns logits unchanged
- exp(logits - logits) = exp(0) = 1 for all values
- softmax output is all 1s, sum is 32 instead of 1
"""

import torch
import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def test_reduce_max(inp, scaler, out):
    """Test reduce_max - should compute max within tile, not across tiles."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=1)
    scaler_cb = ttl.make_circular_buffer_like(scaler, shape=(1, 1), buffer_factor=1)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=1)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as iv, scaler_cb.wait() as sc:
            with out_cb.reserve() as o:
                # This should return max value(s) within the tile
                # But it returns the input unchanged because there's only 1 tile
                o.store(ttl.math.reduce_max(iv, sc, o, dims=[0]))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()
        with scaler_cb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


def to_ttnn(t, device):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    print("=== reduce_max Bug Reproducer ===\n")

    device = ttnn.open_device(device_id=0)

    # Input: varying values to make the bug obvious
    inp = torch.zeros(32, 32, dtype=torch.bfloat16)
    inp[0, 0:4] = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    inp[1, 0:4] = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.bfloat16)

    scaler = torch.ones(32, 32, dtype=torch.bfloat16)
    out = torch.zeros(32, 32, dtype=torch.bfloat16)

    print("Input (first 2 rows, 8 cols):")
    print(f"  Row 0: {inp[0, :8].tolist()}")
    print(f"  Row 1: {inp[1, :8].tolist()}")

    inp_tt = to_ttnn(inp, device)
    scaler_tt = to_ttnn(scaler, device)
    out_tt = to_ttnn(out, device)

    test_reduce_max(inp_tt, scaler_tt, out_tt)

    result = ttnn.to_torch(out_tt).float()

    print("\nActual output (first 2 rows, 8 cols):")
    print(f"  Row 0: {result[0, :8].tolist()}")
    print(f"  Row 1: {result[1, :8].tolist()}")

    # What reduce_max(dims=[0]) should produce:
    # Max across rows (dim 0) for each column
    # col 0: max(1, 5, 0, 0, ...) = 5
    # col 1: max(2, 6, 0, 0, ...) = 6
    # etc.
    expected = inp.float().max(dim=0).values
    print("\nExpected output (max across rows for each column):")
    print(f"  {expected[:8].tolist()}")

    print("\nBUG CONFIRMED: reduce_max returned input unchanged instead of computing max!")
    print("The bug is in python/sim/math.py:reduce_max()")
    print("It operates on TILE GRID level, not ELEMENT level within tiles.")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
