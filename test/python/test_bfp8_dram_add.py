# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# Verify: DRAM interleaved tensors with bfloat8_b data format can be passed
# to a tt-lang kernel and produce correct results.

import torch
import ttnn
import ttl
from utils.correctness import assert_allclose


@ttl.operation(grid=(1, 1))
def add_bfp8_dram(lhs, rhs, out):
    """Add kernel that reads bfloat8_b tensors directly from DRAM."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(2, 2), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        result = l + r
        o.store(result)
        l.pop()
        r.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        tx_lhs = ttl.copy(lhs[0:2, 0:2], lhs_blk)
        tx_lhs.wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        tx_rhs = ttl.copy(rhs[0:2, 0:2], rhs_blk)
        tx_rhs.wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


# CHECK: Testing BFP8 DRAM
print("=== Testing BFP8 DRAM Interleaved Add ===")

device = ttnn.open_device(device_id=0)

try:
    lhs_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((64, 64), 3.0, dtype=torch.bfloat16)
    out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

    # Create DRAM interleaved tensors with bfloat8_b format
    lhs = ttnn.from_torch(
        lhs_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rhs = ttnn.from_torch(
        rhs_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print(f"lhs dtype: {lhs.dtype}")
    # CHECK: bfloat8_b

    add_bfp8_dram(lhs, rhs, out)
    result = ttnn.to_torch(out)

    print(f"\nResult[0,0] = {result[0, 0].item()}")
    print(f"Expected ~5.0")

    # bfp8 has lower precision, use relaxed tolerance
    expected = lhs_torch + rhs_torch
    assert_allclose(result.float(), expected.float(), rtol=0.1, atol=0.5)
    print("\nPASS: BFP8 DRAM interleaved add works!")
    # CHECK: PASS

finally:
    ttnn.close_device(device)

print("\n=== BFP8 DRAM Test Complete ===")
# CHECK: Test Complete
