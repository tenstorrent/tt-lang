# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: TTNN interop with matrix multiplication.
# Based on test_runtime_matmul.py pattern.

import torch
from ttlang.d2m_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== TTNN Interop Matmul Test Complete ===")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)], ttnn_interop=True)
def test_ttnn_interop_matmul(lhs, rhs, out):
    """Matrix multiplication kernel matching test_runtime_matmul.py pattern."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)

    @compute()
    def compute_matmul(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l @ r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_lhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    def dm_rhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()

    return Program(compute_matmul, dm_lhs, dm_rhs)(lhs, rhs, out)


# CHECK: TTNN INTEROP
# CHECK: Found 3 kernels

print("=== Testing TTNN Interop Matmul ===")

device = ttnn.open_device(device_id=0)

try:
    # Use identity matrix for easier verification: I @ A = A
    # Match test_runtime_matmul.py exactly
    lhs_torch = torch.eye(32, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    out_torch = torch.zeros(
        (32, 32), dtype=torch.bfloat16
    )  # Pre-initialize with zeros (issue #31)

    expected = torch.full((32, 32), 2.0, dtype=torch.bfloat16)  # I @ 2s = 2s

    # L1 sharded config
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

    def to_l1_tensor(torch_tensor):
        """Convert torch tensor to TTNN L1 sharded tensor."""
        dram = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.to_memory_config(dram, memory_config=l1_config)

    lhs = to_l1_tensor(lhs_torch)
    rhs = to_l1_tensor(rhs_torch)
    out = to_l1_tensor(out_torch)

    print(f"\nlhs: {lhs.shape} (identity matrix)")
    print(f"rhs: {rhs.shape} (all 2.0)")
    print(f"out: {out.shape} (zeros)")
    print(f"Expected: I @ 2s = all 2.0")

    print("\n=== Running matmul kernel ===")
    test_ttnn_interop_matmul(lhs, rhs, out)

    # Verify
    out_result = ttnn.to_torch(out)

    print("\n=== Results ===")
    print(f"out[0, 0] = {out_result[0, 0].item():.3f}")

    if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: Matmul output matches expected (all 2.0)!")
        # CHECK: PASS
    else:
        print(f"\nFAIL: Expected 2.0, got {out_result[0, 0].item():.3f}")

finally:
    ttnn.close_device(device)

print("\n=== TTNN Interop Matmul Test Complete ===")
# CHECK: TTNN Interop Matmul Test Complete
