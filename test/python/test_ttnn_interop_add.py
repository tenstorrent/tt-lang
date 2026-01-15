# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: TTNN interop path with ttnn.Tensors on device.

import torch
import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def test_ttnn_interop_add(lhs, rhs, out):
    """Simple add kernel compiled for TTNN interop (C++ output)."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Read both inputs - reserve CB and copy directly from tensor to CB
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_out():
        # Write output - wait for data in CB and copy directly from CB to device
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(add_compute, dm_read, dm_out)(lhs, rhs, out)


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

    # Move to L1 with interleaved layout
    print("\nMoving tensors from DRAM to L1 (interleaved)...")
    lhs = ttnn.to_memory_config(lhs_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    rhs = ttnn.to_memory_config(rhs_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = ttnn.to_memory_config(out_dram, memory_config=ttnn.L1_MEMORY_CONFIG)

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
