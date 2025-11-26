# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Verify: Runtime execution with DRAM for large tensors that won't fit in L1.
# L1 is ~1.5MB per core. This test uses 1024x1024 f32 = 4MB which requires DRAM.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_runtime_dram_large(lhs, rhs, out):
    # TensorAccessor() wraps input tensors for DMA with explicit DRAM memory space.
    # Large tensors (4MB each) require DRAM - they won't fit in L1 (~1.5MB per core).
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")
    rhs_accessor = TensorAccessor(rhs, memory_space="DRAM")

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
    def dm_lhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    def dm_rhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


# Minimal MLIR checks - detailed structure verified in test_runtime_dram_add.py
# This test focuses on runtime proof that large tensors work with DRAM.

# CHECK: dram
# CHECK: ttcore.global @lhs
# CHECK: ttcore.global @rhs
# CHECK: func.func @test_runtime_dram_large

# CHECK-LOWERED: #dram = #ttcore.memory_space<dram>
# CHECK-LOWERED: ttcore.global @lhs
# CHECK-LOWERED: ttcore.global @rhs
# CHECK-LOWERED: func.func @test_runtime_dram_large

# Large tensors: 1024x1024 f32 = 4MB each (won't fit in L1)
# Use simple known values for testing: 2 + 3 = 5
lhs = torch.full((1024, 1024), 2.0)
rhs = torch.full((1024, 1024), 3.0)
out = torch.full((1024, 1024), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Tensor size: 1024x1024 f32 = 4MB each (larger than L1 ~1.5MB)")
print(f"lhs: all 2.0")
print(f"rhs: all 3.0")
print(f"out: all -999.0")
print(f"Expected: all 5.0")

test_runtime_dram_large(lhs, rhs, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
print(f"out[512, 512] = {out[512, 512].item()}")
print(
    f"out min/max/mean: {out.min().item():.1f} / {out.max().item():.1f} / {out.mean().item():.1f}"
)

expected = lhs + rhs
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected (2.0 + 3.0 = 5.0)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(
        f"FAIL: Expected all 5.0, got values from {out.min().item()} to {out.max().item()}"
    )
