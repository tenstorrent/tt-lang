# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: DRAM interleaved tensors can be passed directly to tt-lang kernel
# without first moving them to L1. The data movement thread should DMA
# directly from DRAM into CBs.

import torch
import ttnn
import ttl
from test_helpers import to_dram


@ttl.kernel(grid=(1, 1))
def add_dram_direct(lhs, rhs, out):
    """
    Add kernel that reads directly from DRAM interleaved tensors.
    No L1 sharding required - DMA pulls from DRAM into CBs.
    """
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
        # Read from DRAM directly (no L1 intermediate!)
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        # Write result back to DRAM directly
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# CHECK: Testing DRAM Interleaved
print("=== Testing DRAM Interleaved Direct Access ===")

device = ttnn.open_device(device_id=0)

try:
    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch

    # Create DRAM interleaved tensors - NO move to L1!
    lhs = to_dram(lhs_torch, device)
    rhs = to_dram(rhs_torch, device)
    out = to_dram(out_torch, device)

    print(f"lhs memory_config: {lhs.memory_config()}")
    # CHECK: DRAM

    add_dram_direct(lhs, rhs, out)
    result = ttnn.to_torch(out)

    print(f"\nResult[0,0] = {result[0, 0].item()}")
    print(f"Expected[0,0] = {expected[0, 0].item()}")

    if torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: DRAM interleaved direct access works!")
        # CHECK: PASS
    else:
        print(f"\nFAIL: Expected 5.0, got {result[0, 0].item()}")

finally:
    ttnn.close_device(device)

print("\n=== DRAM Interleaved Test Complete ===")
# CHECK: Test Complete
