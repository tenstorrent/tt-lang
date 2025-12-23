# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: DRAM interleaved tensors can be passed directly to tt-lang kernel
# without first moving them to L1. The data movement thread should DMA
# directly from DRAM into CBs.

import torch
from ttlang.ttl_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== DRAM Interleaved Test Complete ===")
    exit(0)


@pykernel_gen(
    grid=(1, 1),
    block_factors=[(1, 1), (1, 1), (1, 1)],
)
def add_dram_direct(lhs, rhs, out):
    """
    Add kernel that reads directly from DRAM interleaved tensors.
    No L1 sharding required - DMA pulls from DRAM into CBs.
    """
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
        # Read from DRAM directly (no L1 intermediate!)
        tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
        tx_lhs.wait()
        tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
        tx_rhs.wait()

    @datamovement()
    def dm_write(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # Write result back to DRAM directly
        tx = copy(out_cb, out_accessor[0, 0])
        tx.wait()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# CHECK: Testing DRAM Interleaved
print("=== Testing DRAM Interleaved Direct Access ===")

device = ttnn.open_device(device_id=0)

try:
    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch  # 5.0

    # Create DRAM interleaved tensors - NO move to L1!
    lhs = ttnn.from_torch(
        lhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Stay in DRAM
    )
    rhs = ttnn.from_torch(
        rhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"lhs memory_config: {lhs.memory_config()}")
    print(f"rhs memory_config: {rhs.memory_config()}")
    print(f"out memory_config: {out.memory_config()}")
    # CHECK: DRAM

    print("\n=== Running kernel with DRAM tensors directly ===")
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
