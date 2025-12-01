# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Runtime test for large DRAM tensors using tilize-on-the-fly with loops.
# 1024x1024 f32 = 4MB (exceeds L1 ~1.5MB per core).
#
# This test demonstrates looping over tiles in both DM and compute threads.
# Processes all 32 tiles along the first row of the tensor.
#
# Flow per iteration:
#   DM read:  DRAM[0, tile_idx] → L1 scalar CB
#   Compute:  tilize → add → untilize → L1 output scalar CB
#   DM write: L1 output scalar CB → DRAM out_dram[0, tile_idx]
#
# Note: out_dram is passed as input (not output) to avoid "output streaming"
# limitation. A dummy L1 output is used to satisfy the framework.

import torch
from ttlang.d2m_api import *

# Process full first row: 1024/32 = 32 tiles
NUM_TILES = 32


# out_dram is DRAM destination (passed as input to avoid streaming output error)
# out_dummy is L1 placeholder output (framework requires an output)
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
def test_runtime_dram_large(lhs, rhs, out_dram, lhs_tiled, rhs_tiled, out_tiled, out_dummy):
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")
    rhs_accessor = TensorAccessor(rhs, memory_space="DRAM")
    out_accessor = TensorAccessor(out_dram, memory_space="DRAM")
    num_tiles = NUM_TILES

    @compute()
    def add_compute(
        lhs_scalar_cb: CircularBuffer,
        rhs_scalar_cb: CircularBuffer,
        out_dram_cb: CircularBuffer,
        lhs_tiled_cb: CircularBuffer,
        rhs_tiled_cb: CircularBuffer,
        out_tiled_cb: CircularBuffer,
        out_dummy_cb: CircularBuffer
    ):
        for tile_idx in range(num_tiles):
            # Wait for scalar data from DM read
            l_scalar = lhs_scalar_cb.wait()
            r_scalar = rhs_scalar_cb.wait()

            # Tilize inputs
            l_tiled = lhs_tiled_cb.reserve()
            r_tiled = rhs_tiled_cb.reserve()
            tilize(l_scalar, l_tiled)
            tilize(r_scalar, r_tiled)
            lhs_tiled_cb.push()
            rhs_tiled_cb.push()
            lhs_scalar_cb.pop()
            rhs_scalar_cb.pop()

            # Compute on tiled data
            l_tile = lhs_tiled_cb.wait()
            r_tile = rhs_tiled_cb.wait()
            o_tiled = out_tiled_cb.reserve()
            result = l_tile + r_tile
            o_tiled.store(result)
            out_tiled_cb.push()
            lhs_tiled_cb.pop()
            rhs_tiled_cb.pop()

            # Untilize result for DM write
            o_tile = out_tiled_cb.wait()
            o_scalar = out_dram_cb.reserve()
            untilize(o_tile, o_scalar)
            out_dram_cb.push()
            out_tiled_cb.pop()

            # Write each result to dummy output CB (used by framework, last value kept)
            dummy = out_dummy_cb.reserve()
            dummy.store(result)
            out_dummy_cb.push()

    @datamovement()
    def dm_read(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
                out_dram_cb: CircularBuffer, lhs_tiled_cb: CircularBuffer,
                rhs_tiled_cb: CircularBuffer, out_tiled_cb: CircularBuffer,
                out_dummy_cb: CircularBuffer):
        for tile_idx in range(num_tiles):
            lhs_shard = lhs_scalar_cb.reserve()
            rhs_shard = rhs_scalar_cb.reserve()
            tx_lhs = dma(lhs_accessor[0, tile_idx], lhs_shard)
            tx_rhs = dma(rhs_accessor[0, tile_idx], rhs_shard)
            tx_lhs.wait()
            tx_rhs.wait()
            lhs_scalar_cb.push()
            rhs_scalar_cb.push()

    @datamovement()
    def dm_write(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
                 out_dram_cb: CircularBuffer, lhs_tiled_cb: CircularBuffer,
                 rhs_tiled_cb: CircularBuffer, out_tiled_cb: CircularBuffer,
                 out_dummy_cb: CircularBuffer):
        for tile_idx in range(num_tiles):
            o_shard = out_dram_cb.wait()
            tx = dma(o_shard, out_accessor[0, tile_idx])
            tx.wait()
            out_dram_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out_dram, lhs_tiled, rhs_tiled, out_tiled, out_dummy)


# Large tensors in DRAM
lhs = torch.full((1024, 1024), 2.0)
rhs = torch.full((1024, 1024), 3.0)
# DRAM output: first row (32 tiles = 32x1024 elements)
out_dram = torch.full((32, 1024), -999.0)
# L1 intermediates (single tile each)
lhs_tiled = torch.zeros((32, 32))
rhs_tiled = torch.zeros((32, 32))
out_tiled = torch.zeros((32, 32))
# Dummy L1 output (framework requires an output argument)
out_dummy = torch.zeros((32, 32))

print("=== BEFORE KERNEL ===")
print(f"Tensor size: 1024x1024 f32 = 4MB (exceeds L1 ~1.5MB)")
print(f"Processing {NUM_TILES} tiles (full first row)")
print(f"lhs: all 2.0, rhs: all 3.0, out_dram: all -999.0")
print(f"Expected: out_dram[0:32, 0:1024] = all 5.0")

test_runtime_dram_large(lhs, rhs, out_dram, lhs_tiled, rhs_tiled, out_tiled, out_dummy)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out_dram[0, 0] = {out_dram[0, 0].item()}")
print(f"out_dram[0, 1023] = {out_dram[0, 1023].item()}")
print(
    f"out_dram min/max/mean: {out_dram.min().item():.1f} / {out_dram.max().item():.1f} / {out_dram.mean().item():.1f}"
)

expected = torch.full((32, 1024), 5.0)
if torch.allclose(out_dram, expected, rtol=1e-2, atol=1e-2):
    print("PASS: All 32 tiles match expected (2.0 + 3.0 = 5.0)")
    # CHECK-OUTPUT: PASS: All 32 tiles match expected
else:
    diff = (out_dram - expected).abs()
    num_wrong = (diff > 0.01).sum().item()
    print(f"FAIL: {num_wrong} elements differ from expected 5.0")
    print(f"  out_dram range: [{out_dram.min().item():.2f}, {out_dram.max().item():.2f}]")
