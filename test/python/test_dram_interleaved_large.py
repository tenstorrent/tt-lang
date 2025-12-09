# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: Large DRAM interleaved tensors work with tile-by-tile processing.
# Unlike the L1 sharded approach, this passes DRAM tensors directly to the kernel.

import torch
from ttlang.d2m_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== DRAM Interleaved Large Test Complete ===")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)], ttnn_interop=True)
def add_tile_kernel_dram(lhs, rhs, out):
    """Add kernel reading directly from DRAM interleaved tensors."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @compute()
    def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_read(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx_lhs = dma(lhs_accessor[0, 0], lhs_shard)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_shard = rhs_cb.reserve()
        tx_rhs = dma(rhs_accessor[0, 0], rhs_shard)
        tx_rhs.wait()
        rhs_cb.push()

    @datamovement()
    def dm_out(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        out_shard = out_cb.wait()
        tx = dma(out_shard, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_out)(lhs, rhs, out)


# CHECK: === DRAM Interleaved Large Test ===
print("=== DRAM Interleaved Large Test ===")

device = ttnn.open_device(device_id=0)

try:
    # Large tensor: 1024x1024 = 32x32 tiles of 32x32 each
    # Total: 3 tensors * ~2MB = ~6MB (doesn't fit in L1)
    H, W = 1024, 1024
    TILE_H, TILE_W = 32, 32
    num_tiles_h = H // TILE_H
    num_tiles_w = W // TILE_W
    total_tiles = num_tiles_h * num_tiles_w

    size_mb = H * W * 2 / (1024 * 1024)
    print(f"Tensor size: {H}x{W} = {size_mb:.1f}MB per tensor ({size_mb*3:.1f}MB total)")
    print(f"Tile grid: {num_tiles_h}x{num_tiles_w} = {total_tiles} tiles")

    # Create input tensors with distinct values per tile
    lhs_torch = torch.zeros((H, W), dtype=torch.bfloat16)
    rhs_torch = torch.zeros((H, W), dtype=torch.bfloat16)

    for ti in range(num_tiles_h):
        for tj in range(num_tiles_w):
            tile_val = float(ti * num_tiles_w + tj)
            lhs_torch[ti*TILE_H:(ti+1)*TILE_H, tj*TILE_W:(tj+1)*TILE_W] = tile_val
            rhs_torch[ti*TILE_H:(ti+1)*TILE_H, tj*TILE_W:(tj+1)*TILE_W] = 1.0

    expected = lhs_torch + rhs_torch
    out_torch = torch.full((H, W), -999.0, dtype=torch.bfloat16)

    # Create large DRAM interleaved tensors
    print("\nCreating large tensors in DRAM (interleaved)...")
    lhs_dram = ttnn.from_torch(
        lhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rhs_dram = ttnn.from_torch(
        rhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_dram = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"lhs memory_config: {lhs_dram.memory_config()}")
    print(f"DRAM tensors created: lhs={lhs_dram.shape}, rhs={rhs_dram.shape}, out={out_dram.shape}")
    # CHECK: DRAM

    # Compile kernel once using single-tile slices as samples
    print("\nCompiling kernel (once)...")
    # CHECK: Compiling kernel

    # Get first tile to use as sample for compilation
    sample_lhs = ttnn.slice(lhs_dram, [0, 0], [TILE_H, TILE_W])
    sample_rhs = ttnn.slice(rhs_dram, [0, 0], [TILE_H, TILE_W])
    sample_out = ttnn.slice(out_dram, [0, 0], [TILE_H, TILE_W])

    print(f"Sample tile lhs memory_config: {sample_lhs.memory_config()}")

    # Compile and get reusable kernel
    kernel = add_tile_kernel_dram.compile(sample_lhs, sample_rhs, sample_out)
    print(f"Kernel compiled! Type: {type(kernel)}")

    # Process tile by tile - directly from DRAM!
    print(f"\nProcessing {total_tiles} tiles directly from DRAM...")
    # CHECK: Processing 1024 tiles

    for ti in range(num_tiles_h):
        for tj in range(num_tiles_w):
            tile_idx = ti * num_tiles_w + tj

            row_start, row_end = ti * TILE_H, (ti + 1) * TILE_H
            col_start, col_end = tj * TILE_W, (tj + 1) * TILE_W

            # Slice tiles directly from DRAM - NO L1 intermediate!
            lhs_tile = ttnn.slice(lhs_dram, [row_start, col_start], [row_end, col_end])
            rhs_tile = ttnn.slice(rhs_dram, [row_start, col_start], [row_end, col_end])
            out_tile = ttnn.slice(out_dram, [row_start, col_start], [row_end, col_end])

            # Run kernel directly on DRAM tiles
            kernel(lhs_tile, rhs_tile, out_tile)

            # Get result as torch for verification
            out_tile_torch = ttnn.to_torch(out_tile)

            # Store in our output tensor
            out_torch[row_start:row_end, col_start:col_end] = out_tile_torch

            if tile_idx % 100 == 0:
                print(f"  Processed tile {tile_idx}/{total_tiles}")

    print(f"\nAll {total_tiles} tiles processed!")
    # CHECK: All 1024 tiles processed

    # Verify results
    print("\n=== Verification ===")
    print(f"Expected corner values: [0,0]={expected[0,0].item()}, [0,32]={expected[0,32].item()}, [32,0]={expected[32,0].item()}")
    print(f"Got corner values:      [0,0]={out_torch[0,0].item()}, [0,32]={out_torch[0,32].item()}, [32,0]={out_torch[32,0].item()}")

    if torch.allclose(out_torch.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: All tiles computed correctly from DRAM interleaved!")
        # CHECK: PASS
    else:
        diff = (out_torch.float() - expected.float()).abs()
        max_err = diff.max().item()
        err_loc = (diff == max_err).nonzero()[0]
        print(f"\nFAIL: Max error = {max_err} at {err_loc.tolist()}")
        print(f"  Expected: {expected[err_loc[0], err_loc[1]].item()}")
        print(f"  Got: {out_torch[err_loc[0], err_loc[1]].item()}")

finally:
    ttnn.close_device(device)

print("\n=== DRAM Interleaved Large Test Complete ===")
# CHECK: DRAM Interleaved Large Test Complete
