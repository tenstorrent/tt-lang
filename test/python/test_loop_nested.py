# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: Nested loops - iterate over 2x2 tile grid (4 iterations total).
# Tests nested for loops in compute and datamovement threads.

import torch
from ttlang.ttl_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== Nested Loop Test Complete ===")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)], ttnn_interop=True)
def test_nested_loop_add(lhs, rhs, out):
    """Add kernel with nested loops - 2x2 = 4 iterations."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @compute()
    def nested_add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        for i in range(2):
            for j in range(2):
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
        for i in range(2):
            for j in range(2):
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
        for i in range(2):
            for j in range(2):
                out_shard = out_cb.wait()
                tx = dma(out_shard, out_accessor[0, 0])
                tx.wait()
                out_cb.pop()

    return Program(nested_add_compute, dm_read, dm_out)(lhs, rhs, out)


# CHECK: === Nested Loop Test ===
print("=== Nested Loop Test ===")

device = ttnn.open_device(device_id=0)

try:
    # Create torch tensors
    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 5.0, dtype=torch.bfloat16)
    out_torch = torch.full((32, 32), -999.0, dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch  # 7.0

    # Create DRAM tensors
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

    # Move to L1 sharded
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        (32, 32),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    lhs = ttnn.to_memory_config(lhs_dram, memory_config=l1_config)
    rhs = ttnn.to_memory_config(rhs_dram, memory_config=l1_config)
    out = ttnn.to_memory_config(out_dram, memory_config=l1_config)

    print(f"Running nested loop kernel (2x2 = 4 iterations)...")
    print(f"  lhs: all 2.0, rhs: all 5.0")
    print(f"  Expected: all 7.0")

    test_nested_loop_add(lhs, rhs, out)

    # Get result
    out_result = ttnn.to_torch(out)

    print(f"\n=== AFTER KERNEL ===")
    print(f"out[0,0] = {out_result[0,0].item()}")
    print(
        f"out min/max/mean: {out_result.min().item():.1f} / {out_result.max().item():.1f} / {out_result.float().mean().item():.1f}"
    )

    if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: Nested loop test completed (2 + 5 = 7, 4 iterations)")
        # CHECK: PASS
    else:
        print(
            f"\nFAIL: Expected all 7.0, got {out_result.min().item()} to {out_result.max().item()}"
        )

finally:
    ttnn.close_device(device)

print("\n=== Nested Loop Test Complete ===")
# CHECK: Nested Loop Test Complete
