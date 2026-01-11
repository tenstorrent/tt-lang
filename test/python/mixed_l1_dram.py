# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

"""
Test mixed L1/DRAM tensor configurations.

This test verifies that TensorAccessorArgs correctly indexes into
compile_time_args when tensors have different memory configurations.

Configuration tested:
- lhs: L1 (interleaved)
- rhs: L1 (interleaved)
- out: DRAM (interleaved)
"""

import torch
import ttnn
from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program


@ttl.kernel(grid=(1, 1))
def add_mixed_memory(lhs, rhs, out):
    """Add kernel with mixed L1 inputs and DRAM output."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
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
        # Read from L1 tensors
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute, dm_read, dm_write)(lhs, rhs, out)


# CHECK: === Mixed L1/DRAM Test ===
print("=== Mixed L1/DRAM Test ===")

device = ttnn.open_device(device_id=0)

try:
    # Create test tensors
    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.full((32, 32), -1000.0, dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch  # Should be 5.0

    # Create DRAM tensors first
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

    # Move inputs to L1, keep output in DRAM
    # This creates the mixed memory configuration that exposes the bug
    lhs = ttnn.to_memory_config(lhs_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    rhs = ttnn.to_memory_config(rhs_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = out_dram  # Intentionally keep in DRAM

    print(f"lhs: {lhs.memory_config()}")
    print(f"rhs: {rhs.memory_config()}")
    print(f"out: {out.memory_config()}")

    # CHECK: Running kernel
    print("Running kernel...")
    add_mixed_memory(lhs, rhs, out)

    # Verify results
    out_result = ttnn.to_torch(out)

    print(f"\nout[0:3, 0:3] =\n{out_result[0:3, 0:3]}")
    print(f"expected[0:3, 0:3] =\n{expected[0:3, 0:3]}")

    if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        # CHECK: PASS
        print("\nPASS: Mixed L1/DRAM test passed!")
    else:
        max_err = (out_result.float() - expected.float()).abs().max().item()
        print(f"\nFAIL: Max error = {max_err:.6f}")
        print("This failure indicates TensorAccessorArgs is using wrong CTA index")

finally:
    ttnn.close_device(device)

# CHECK: Test Complete
print("\n=== Test Complete ===")
