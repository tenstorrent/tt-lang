# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: Process large DRAM tensor tile-by-tile via ttnn + tt-lang kernel

import torch
from ttlang.d2m_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== Tiled Loop Test Complete ===")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)], ttnn_interop=True)
def add_tile_kernel(lhs, rhs, out):
    """Add kernel for a single tile."""
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

        rhs_shard = rhs_cb.reserve()
        tx_rhs = dma(rhs_accessor[0, 0], rhs_shard)
        tx_rhs.wait()

    @datamovement()
    def dm_out(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        out_shard = out_cb.wait()
        tx = dma(out_shard, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_out)(lhs, rhs, out)


def create_sharded_l1_config(device):
    """Create L1 sharded memory config for a single 32x32 tile on core (0,0)."""
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        (32, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )


# CHECK: === Tiled Loop Test ===
print("=== Tiled Loop Test ===")

device = ttnn.open_device(device_id=0)

try:
    # Large tensor: 1024x1024 = 32x32 tiles of 32x32 each
    # Size: 1024*1024*2 bytes = 2MB per tensor (> 1.5MB L1 limit)
    # Total: 3 tensors * 2MB = 6MB (definitely won't fit in L1)
    H, W = 1024, 1024
    TILE_H, TILE_W = 32, 32
    num_tiles_h = H // TILE_H
    num_tiles_w = W // TILE_W
    total_tiles = num_tiles_h * num_tiles_w

    size_mb = H * W * 2 / (1024 * 1024)
    print(
        f"Tensor size: {H}x{W} = {size_mb:.1f}MB per tensor ({size_mb*3:.1f}MB total)"
    )
    print(f"Tile grid: {num_tiles_h}x{num_tiles_w} = {total_tiles} tiles")

    # Create input tensors with distinct values per tile for easy verification
    lhs_torch = torch.zeros((H, W), dtype=torch.bfloat16)
    rhs_torch = torch.zeros((H, W), dtype=torch.bfloat16)

    # Fill each tile with its tile index for lhs, and 1.0 for rhs
    for ti in range(num_tiles_h):
        for tj in range(num_tiles_w):
            tile_val = float(ti * num_tiles_w + tj)  # 0, 1, 2, ... 15
            lhs_torch[
                ti * TILE_H : (ti + 1) * TILE_H, tj * TILE_W : (tj + 1) * TILE_W
            ] = tile_val
            rhs_torch[
                ti * TILE_H : (ti + 1) * TILE_H, tj * TILE_W : (tj + 1) * TILE_W
            ] = 1.0

    expected = lhs_torch + rhs_torch
    out_torch = torch.full((H, W), -999.0, dtype=torch.bfloat16)

    # Create DRAM tensors
    print("\nCreating large tensors in DRAM...")
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

    print(
        f"DRAM tensors: lhs={lhs_dram.shape}, rhs={rhs_dram.shape}, out={out_dram.shape}"
    )

    l1_config = create_sharded_l1_config(device)

    # Compile kernel ONCE using the first tile as sample
    print("\nCompiling kernel (once)...")
    # CHECK: Compiling kernel

    # Get first tile to use as sample for compilation
    sample_lhs = ttnn.to_memory_config(
        ttnn.slice(lhs_dram, [0, 0], [TILE_H, TILE_W]), memory_config=l1_config
    )
    sample_rhs = ttnn.to_memory_config(
        ttnn.slice(rhs_dram, [0, 0], [TILE_H, TILE_W]), memory_config=l1_config
    )
    sample_out = ttnn.to_memory_config(
        ttnn.slice(out_dram, [0, 0], [TILE_H, TILE_W]), memory_config=l1_config
    )

    # Compile and get reusable kernel
    kernel = add_tile_kernel.compile(sample_lhs, sample_rhs, sample_out)
    print(f"Kernel compiled! Type: {type(kernel)}")

    # Process tile by tile using compiled kernel
    print(f"\nProcessing {total_tiles} tiles...")
    # CHECK: Processing 1024 tiles

    for ti in range(num_tiles_h):
        for tj in range(num_tiles_w):
            tile_idx = ti * num_tiles_w + tj

            # Slice out the tile from DRAM tensors
            row_start, row_end = ti * TILE_H, (ti + 1) * TILE_H
            col_start, col_end = tj * TILE_W, (tj + 1) * TILE_W

            # Use ttnn.slice to extract tile (returns new tensor)
            lhs_tile = ttnn.slice(lhs_dram, [row_start, col_start], [row_end, col_end])
            rhs_tile = ttnn.slice(rhs_dram, [row_start, col_start], [row_end, col_end])
            out_tile = ttnn.slice(out_dram, [row_start, col_start], [row_end, col_end])

            # Move tiles to sharded L1
            lhs_l1 = ttnn.to_memory_config(lhs_tile, memory_config=l1_config)
            rhs_l1 = ttnn.to_memory_config(rhs_tile, memory_config=l1_config)
            out_l1 = ttnn.to_memory_config(out_tile, memory_config=l1_config)

            # Run compiled kernel on this tile (no recompilation!)
            kernel(lhs_l1, rhs_l1, out_l1)

            # Copy result back to the output DRAM tensor
            # First move result back to DRAM
            out_tile_result = ttnn.to_memory_config(
                out_l1, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # Get result as torch for verification
            out_tile_torch = ttnn.to_torch(out_tile_result)

            # Store in our output tensor
            out_torch[row_start:row_end, col_start:col_end] = out_tile_torch

            # Progress indicator (every 100 tiles)
            if tile_idx % 100 == 0:
                print(f"  Processed tile {tile_idx}/{total_tiles}")

    print(f"\nAll {total_tiles} tiles processed!")
    # CHECK: All 1024 tiles processed

    # Verify results
    print("\n=== Verification ===")
    print(
        f"Expected corner values: [0,0]={expected[0,0].item()}, [0,32]={expected[0,32].item()}, [32,0]={expected[32,0].item()}"
    )
    print(
        f"Got corner values:      [0,0]={out_torch[0,0].item()}, [0,32]={out_torch[0,32].item()}, [32,0]={out_torch[32,0].item()}"
    )

    if torch.allclose(out_torch.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: All tiles computed correctly!")
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

print("\n=== Tiled Loop Test Complete ===")
# CHECK: Tiled Loop Test Complete
