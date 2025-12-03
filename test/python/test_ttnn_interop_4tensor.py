# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: TTNN interop with 4 tensors (3 inputs + 1 output).
# Tests that more than 3 tensors works correctly.

import torch
from ttlang.d2m_api import *

try:
    import ttnn
except ImportError:
    print("TTNN not available - this test requires ttnn")
    print("=== TTNN Interop 4-Tensor Test Complete ===")
    exit(0)


@pykernel_gen(
    grid=(1, 1),
    block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)],
    ttnn_interop=True,
)
def test_ttnn_interop_sum3(a, b, c, out):
    """Sum 3 input tensors: (a + b) + c."""
    a_accessor = TensorAccessor(a)
    b_accessor = TensorAccessor(b)
    c_accessor = TensorAccessor(c)

    @compute()
    def sum3_compute(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        av = a_cb.wait()
        bv = b_cb.wait()
        cv = c_cb.wait()
        o = out_cb.reserve()
        # Sum: (a + b) + c
        result = (av + bv) + cv
        o.store(result)
        a_cb.pop()
        b_cb.pop()
        c_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_a(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        a_shard = a_cb.reserve()
        tx = dma(a_accessor[0, 0], a_shard)
        tx.wait()

    @datamovement()
    def dm_b(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        b_shard = b_cb.reserve()
        tx = dma(b_accessor[0, 0], b_shard)
        tx.wait()

    @datamovement()
    def dm_c(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        c_shard = c_cb.reserve()
        tx = dma(c_accessor[0, 0], c_shard)
        tx.wait()

    return Program(sum3_compute, dm_a, dm_b, dm_c)(a, b, c, out)


# CHECK: TTNN INTEROP
# CHECK: Found 4 kernels

print("=== Testing TTNN Interop 4-Tensor Sum ===")

device = ttnn.open_device(device_id=0)

try:
    # Create 3 input tensors with values 1, 2, 3
    a_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
    b_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    c_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # Expected: 1 + 2 + 3 = 6
    expected = a_torch + b_torch + c_torch

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

    # Convert all tensors
    a = to_l1_tensor(a_torch)
    b = to_l1_tensor(b_torch)
    c = to_l1_tensor(c_torch)
    out = to_l1_tensor(out_torch)

    print(f"\n3 inputs + 1 output created, each {a.shape}")
    print(f"Input values: a=1, b=2, c=3")
    print(f"Expected sum: 6")

    print("\n=== Running 3-input sum kernel ===")
    test_ttnn_interop_sum3(a, b, c, out)

    # Verify
    out_result = ttnn.to_torch(out)

    print("\n=== Results ===")
    print(f"Output[0,0] = {out_result[0,0].item()}")
    print(f"Expected[0,0] = {expected[0,0].item()}")

    if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("\nPASS: 3-input sum matches expected!")
        # CHECK: PASS
    else:
        max_err = (out_result.float() - expected.float()).abs().max().item()
        print(f"\nFAIL: Max error = {max_err:.6f}")

finally:
    ttnn.close_device(device)

print("\n=== TTNN Interop 4-Tensor Test Complete ===")
# CHECK: TTNN Interop 4-Tensor Test Complete
