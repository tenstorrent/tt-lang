# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: TTNN interop path with ttnn.Tensors on device.

import os
import platform
import torch
from ttlang.ttl_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== TTNN Interop Test Complete ===")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)], ttnn_interop=True)
def test_ttnn_interop_add(lhs, rhs, out):
    """Simple add kernel compiled for TTNN interop (C++ output)."""
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
        # Read both inputs - reserve CB and copy directly from accessor to CB
        lhs_cb.reserve()
        tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
        tx_lhs.wait()

        rhs_cb.reserve()
        tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
        tx_rhs.wait()

    @datamovement()
    def dm_out(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        # Write output - wait for data in CB and copy directly from CB to device
        out_cb.wait()
        tx = copy(out_cb, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_out)(lhs, rhs, out)


# CHECK: TTNN INTEROP
# CHECK: Created ProgramDescriptor

print("=== Testing TTNN Interop Path ===")
print("Opening device and creating ttnn.Tensors...")

# Open device
device = ttnn.open_device(device_id=0)

try:
    # Create torch tensors first
    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.full((32, 32), -999.0, dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch

    # Convert to TTNN tensors on device - start in DRAM, then move to L1
    # This avoids bank-aware DRAM addressing complexity for now
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

    # Move to L1 with SHARDED layout on a single core
    # This ensures we know exactly which core has the data
    print("\nMoving tensors from DRAM to L1 (sharded on single core)...")
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        (32, 32),  # shard shape = full tensor
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    l1_sharded_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )
    lhs = ttnn.to_memory_config(lhs_dram, memory_config=l1_sharded_config)
    rhs = ttnn.to_memory_config(rhs_dram, memory_config=l1_sharded_config)
    out = ttnn.to_memory_config(out_dram, memory_config=l1_sharded_config)

    print(f"\nttnn.Tensors in L1:")
    print(f"  lhs: {lhs.shape}, dtype={lhs.dtype}, memory_config={lhs.memory_config()}")
    print(f"  rhs: {rhs.shape}, dtype={rhs.dtype}, memory_config={rhs.memory_config()}")
    print(f"  out: {out.shape}, dtype={out.dtype}, memory_config={out.memory_config()}")

    print("\n=== Running tt-lang kernel with ttnn.Tensors ===")
    test_ttnn_interop_add(lhs, rhs, out)

    # Copy result back to host for verification
    out_result = ttnn.to_torch(out)

    print("\n=== AFTER KERNEL ===")
    print(f"out[0:3, 0:3] =\n{out_result[0:3, 0:3]}")
    print(f"expected[0:3, 0:3] =\n{expected[0:3, 0:3]}")

    # Verify results
    if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: Output matches expected!")
        # CHECK: PASS
    else:
        max_err = (out_result.float() - expected.float()).abs().max().item()
        print(f"\nFAIL: Max error = {max_err:.6f}")

finally:
    ttnn.close_device(device)

print("\n=== TTNN Interop Test Complete ===")
# CHECK: TTNN Interop Test Complete
