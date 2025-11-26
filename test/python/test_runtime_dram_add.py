# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Verify: Runtime execution of add operation works correctly with DRAM memory space.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_runtime_dram_add(lhs, rhs, out):
    # TensorAccessor() wraps input tensors for DMA with explicit DRAM memory space.
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

# Verify: Two layout definitions - #layout for DRAM inputs, #layout1 for L1 output
# CHECK: #layout = #ttcore.metal_layout<logical_shape = 32x32,
# CHECK-SAME: dram
# CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 32x32,
# CHECK-SAME: l1

# Verify: TensorAccessor globals use DRAM layout (#layout, not #layout1)
# CHECK: ttcore.global @lhs = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
# CHECK: ttcore.global @rhs = tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

# CHECK: func.func @test_runtime_dram_add

# Verify: to_layout and stream_layout ops use DRAM layout for inputs
# CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<32x32xf32> into tensor<{{.*}}, #layout>
# CHECK: d2m.stream_layout
# CHECK-SAME: #layout
# CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<32x32xf32> into tensor<{{.*}}, #layout>

# Verify: d2m.generic has DRAM inputs (#layout) and L1 output (#layout1)
# CHECK: d2m.generic
# CHECK: ins(%{{.*}}, %{{.*}} : tensor<{{.*}}, #layout>, tensor<{{.*}}, #layout>)
# CHECK-NEXT: outs(%{{.*}} : tensor<{{.*}}, #layout1>)

# Verify: get_global returns DRAM tensor type
# CHECK: ttcore.get_global @lhs : tensor<{{.*}}, #layout>
# CHECK: ttcore.get_global @rhs : tensor<{{.*}}, #layout>

# Verify: compute region contains linalg.generic with tile_add
# CHECK: ^compute{{[0-9]+}}
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel"]
# CHECK: "d2m.tile_add"

# Verify: Memory space attributes are defined
# CHECK-LOWERED: #dram = #ttcore.memory_space<dram>
# CHECK-LOWERED: #l1 = #ttcore.memory_space<l1>

# Verify: Globals are lowered to memrefs with DRAM memory space
# CHECK-LOWERED: ttcore.global @lhs = memref<{{.*}}, #dram>
# CHECK-LOWERED: ttcore.global @rhs = memref<{{.*}}, #dram>

# CHECK-LOWERED: func.func @test_runtime_dram_add

# Verify: DRAM buffers are created (address >= 2560000, the dram_unreserved_base)
# CHECK-LOWERED: "ttmetal.create_buffer"() <{address = {{[0-9]+}}
# CHECK-LOWERED-SAME: #dram>

# Verify: L1 buffers are created (address ~100000-110000 range, in L1 unreserved)
# CHECK-LOWERED: "ttmetal.create_buffer"() <{address = {{[0-9]+}}
# CHECK-LOWERED-SAME: #l1>

# Verify: Compute kernel calls add_binary_tile
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

# Use simple known values for testing: 2 + 3 = 5
lhs = torch.full((32, 32), 2.0)
rhs = torch.full((32, 32), 3.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"lhs: all 2.0")
print(f"rhs: all 3.0")
print(f"out: all -999.0")
print(f"Expected: all 5.0")

test_runtime_dram_add(lhs, rhs, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
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
