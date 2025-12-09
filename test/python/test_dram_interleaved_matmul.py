# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: DRAM interleaved tensors work with matrix multiplication.
# DMAs read directly from DRAM into CBs without L1 intermediate.

import torch
from ttlang.d2m_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== DRAM Interleaved Matmul Test Complete ===")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)], ttnn_interop=True)
def matmul_dram_direct(lhs, rhs, out):
    """Matrix multiplication reading directly from DRAM interleaved tensors."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @compute()
    def compute_matmul(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l @ r
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
    def dm_write(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        out_shard = out_cb.wait()
        tx = dma(out_shard, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_matmul, dm_read, dm_write)(lhs, rhs, out)


# CHECK: Testing DRAM Interleaved Matmul
print("=== Testing DRAM Interleaved Matmul ===")

device = ttnn.open_device(device_id=0)

try:
    # Identity matrix test: I @ A = A
    lhs_torch = torch.eye(32, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    expected = torch.full((32, 32), 2.0, dtype=torch.bfloat16)  # I @ 2s = 2s

    # Create DRAM interleaved tensors - NO L1 intermediate!
    lhs = ttnn.from_torch(
        lhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    print(f"\nlhs: {lhs.shape} (identity matrix)")
    print(f"rhs: {rhs.shape} (all 2.0)")
    print(f"out: {out.shape} (zeros)")
    print(f"Expected: I @ 2s = all 2.0")

    print("\n=== Running matmul kernel with DRAM tensors directly ===")
    matmul_dram_direct(lhs, rhs, out)

    result = ttnn.to_torch(out)

    print("\n=== Results ===")
    print(f"out[0, 0] = {result[0, 0].item():.3f}")

    if torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: DRAM interleaved matmul works!")
        # CHECK: PASS: DRAM interleaved matmul
    else:
        print(f"\nFAIL: Expected 2.0, got {result[0, 0].item():.3f}")

finally:
    ttnn.close_device(device)

print("\n=== DRAM Interleaved Matmul Test Complete ===")
# CHECK: Test Complete
