# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Runtime test for large DRAM tensors using tilize-on-the-fly with loops.
# 1024x1024 f32 = 4MB (exceeds L1 ~1.5MB per core).
#
# This test demonstrates looping over tiles in both DM and compute threads.
# Processes NUM_TILES tiles along the first row of the tensor.

import torch
from ttlang.d2m_api import *

# Number of tiles to process (along first row)
NUM_TILES = 4


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
def test_runtime_dram_large(lhs, rhs, lhs_tiled, rhs_tiled, out):
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")
    rhs_accessor = TensorAccessor(rhs, memory_space="DRAM")
    # Capture NUM_TILES in closure for use in kernel threads
    num_tiles = NUM_TILES

    @compute()
    def add_compute(
        lhs_scalar_cb: CircularBuffer,
        rhs_scalar_cb: CircularBuffer,
        lhs_tiled_cb: CircularBuffer,
        rhs_tiled_cb: CircularBuffer,
        out_cb: CircularBuffer
    ):
        # Loop over num_tiles tiles
        for tile_idx in range(num_tiles):
            # Wait for scalar data from DM
            l_scalar = lhs_scalar_cb.wait()
            r_scalar = rhs_scalar_cb.wait()

            # Reserve tiled buffers and tilize
            l_tiled = lhs_tiled_cb.reserve()
            r_tiled = rhs_tiled_cb.reserve()

            tilize(l_scalar, l_tiled)
            tilize(r_scalar, r_tiled)

            lhs_tiled_cb.push()
            rhs_tiled_cb.push()
            lhs_scalar_cb.pop()
            rhs_scalar_cb.pop()

            # Wait for tiled data and compute
            l_tile = lhs_tiled_cb.wait()
            r_tile = rhs_tiled_cb.wait()

            o = out_cb.reserve()
            result = l_tile + r_tile
            o.store(result)
            out_cb.push()

            lhs_tiled_cb.pop()
            rhs_tiled_cb.pop()

    @datamovement()
    def dm_lhs(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
               lhs_tiled_cb: CircularBuffer, rhs_tiled_cb: CircularBuffer,
               out_cb: CircularBuffer):
        # Loop over num_tiles tiles along first row
        for tile_idx in range(num_tiles):
            lhs_shard = lhs_scalar_cb.reserve()
            # DMA tile[0, tile_idx] from DRAM
            tx = dma(lhs_accessor[0, tile_idx], lhs_shard)
            tx.wait()
            lhs_scalar_cb.push()

    @datamovement()
    def dm_rhs(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
               lhs_tiled_cb: CircularBuffer, rhs_tiled_cb: CircularBuffer,
               out_cb: CircularBuffer):
        # Loop over num_tiles tiles along first row
        for tile_idx in range(num_tiles):
            rhs_shard = rhs_scalar_cb.reserve()
            # DMA tile[0, tile_idx] from DRAM
            tx = dma(rhs_accessor[0, tile_idx], rhs_shard)
            tx.wait()
            rhs_scalar_cb.push()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, lhs_tiled, rhs_tiled, out)


lhs = torch.full((1024, 1024), 2.0)
rhs = torch.full((1024, 1024), 3.0)
lhs_tiled = torch.zeros((32, 32))
rhs_tiled = torch.zeros((32, 32))
# Output receives the last tile computed (tile[0, NUM_TILES-1])
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Tensor size: 1024x1024 f32 = 4MB (exceeds L1 ~1.5MB)")
print(f"Processing {NUM_TILES} tiles along first row")
print(f"lhs: all 2.0, rhs: all 3.0, out: all -999.0")
print(f"Expected: all 5.0 (last tile result)")

test_runtime_dram_large(lhs, rhs, lhs_tiled, rhs_tiled, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
print(
    f"out min/max/mean: {out.min().item():.1f} / {out.max().item():.1f} / {out.mean().item():.1f}"
)

expected = torch.full((32, 32), 5.0)
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected (2.0 + 3.0 = 5.0)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(
        f"FAIL: Expected all 5.0, got values from {out.min().item()} to {out.max().item()}"
    )
