# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Verify: Runtime execution of add operation works correctly with DRAM memory space.
# Uses 5 CBs for tilize-on-the-fly:
#   - 2 scalar input CBs (from DRAM)
#   - 2 tiled intermediate CBs (tilize destinations)
#   - 1 tiled output CB

import torch
from ttlang.d2m_api import *


# num_outs=1: only 'out' is an output
# lhs, rhs are DRAM streams; lhs_tiled, rhs_tiled are empty L1 buffers for tilize
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)], num_outs=1)
def test_runtime_dram_add(lhs, rhs, lhs_tiled, rhs_tiled, out):
    # TensorAccessor() wraps input tensors for DMA with explicit DRAM memory space.
    # Scalar data is staged to DRAM, then pulled to L1 and tilized on-the-fly in compute.
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")
    rhs_accessor = TensorAccessor(rhs, memory_space="DRAM")

    @compute()
    def add_compute(
        lhs_scalar_cb: CircularBuffer,  # ins[0] - scalar CB (from DRAM DMA)
        rhs_scalar_cb: CircularBuffer,  # ins[1] - scalar CB (from DRAM DMA)
        lhs_tiled_cb: CircularBuffer,   # ins[2] - empty L1 buffer (tilize destination)
        rhs_tiled_cb: CircularBuffer,   # ins[3] - empty L1 buffer (tilize destination)
        out_cb: CircularBuffer          # outs[0] - tiled CB (final output)
    ):
        # 1. Wait for scalar data from DRAM-backed CBs (input pattern)
        l_scalar = lhs_scalar_cb.wait()
        r_scalar = rhs_scalar_cb.wait()

        # 2. Reserve tiled CBs for tilize output (output pattern)
        l_tiled = lhs_tiled_cb.reserve()
        r_tiled = rhs_tiled_cb.reserve()

        # 3. Tilize: scalar → tiled
        tilize(l_scalar, l_tiled)
        tilize(r_scalar, r_tiled)

        # 4. Push tiled data (make available), pop scalar data (done with it)
        lhs_tiled_cb.push()
        rhs_tiled_cb.push()
        lhs_scalar_cb.pop()
        rhs_scalar_cb.pop()

        # 5. Wait for tiled data we just pushed (immediate since same thread)
        l_tile = lhs_tiled_cb.wait()
        r_tile = rhs_tiled_cb.wait()

        # 6. Reserve output, perform add, store result
        o = out_cb.reserve()
        result = l_tile + r_tile
        o.store(result)
        out_cb.push()

        # 7. Pop tiled CBs
        lhs_tiled_cb.pop()
        rhs_tiled_cb.pop()

    @datamovement()
    def dm_lhs(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
               lhs_tiled_cb: CircularBuffer, rhs_tiled_cb: CircularBuffer,
               out_cb: CircularBuffer):
        lhs_shard = lhs_scalar_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()
        lhs_scalar_cb.push()

    @datamovement()
    def dm_rhs(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
               lhs_tiled_cb: CircularBuffer, rhs_tiled_cb: CircularBuffer,
               out_cb: CircularBuffer):
        rhs_shard = rhs_scalar_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()
        rhs_scalar_cb.push()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, lhs_tiled, rhs_tiled, out)

# Verify: Two layout definitions - #layout for DRAM, #layout1 for L1
# CHECK: #layout = #ttcore.metal_layout<logical_shape = 32x32,
# CHECK-SAME: dram
# CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 32x32,
# CHECK-SAME: l1

# Verify: TensorAccessor globals use DRAM layout
# CHECK: ttcore.global @lhs = tensor<{{.*}}, #layout>
# CHECK: ttcore.global @rhs = tensor<{{.*}}, #layout>

# CHECK: func.func @test_runtime_dram_add

# Verify: to_layout stages host tensor to DRAM
# CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<32x32xf32> into tensor<{{.*}}, #layout>
# CHECK: d2m.stream_layout
# CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<32x32xf32> into tensor<{{.*}}, #layout>

# Verify: d2m.generic has 4 inputs (2 DRAM + 2 L1 buffers) and 1 L1 output
# CHECK: d2m.generic
# CHECK: ins({{.*}} : tensor<{{.*}}, #layout>, tensor<{{.*}}, #layout>, tensor<{{.*}}, #layout1>, tensor<{{.*}}, #layout1>)
# CHECK: outs({{.*}} : tensor<{{.*}}, #layout1>)

# Verify: get_global returns DRAM tensor type (used for DMA source)
# CHECK: ttcore.get_global @lhs : tensor<{{.*}}, #layout>
# CHECK: ttcore.get_global @rhs : tensor<{{.*}}, #layout>

# Verify: compute region contains tile_tilize_block and linalg.generic with tile_add
# CHECK: ^compute{{[0-9]+}}
# CHECK: d2m.tile_tilize_block
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

# Verify: DRAM buffers are created
# CHECK-LOWERED: "ttmetal.create_buffer"() <{address = {{[0-9]+}}
# CHECK-LOWERED-SAME: #dram>

# Verify: L1 buffers are created
# CHECK-LOWERED: "ttmetal.create_buffer"() <{address = {{[0-9]+}}
# CHECK-LOWERED-SAME: #l1>

# Verify: Compute kernel calls tilize and add_binary_tile
# CHECK-LOWERED: emitc.call_opaque "tilize_block"
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

# Use simple known values for testing: 2 + 3 = 5
lhs = torch.full((32, 32), 2.0)
rhs = torch.full((32, 32), 3.0)
lhs_tiled = torch.zeros((32, 32))  # intermediate tiled buffer (stays on device)
rhs_tiled = torch.zeros((32, 32))  # intermediate tiled buffer (stays on device)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"lhs: all 2.0")
print(f"rhs: all 3.0")
print(f"out: all -999.0")
print(f"Expected: all 5.0")

test_runtime_dram_add(lhs, rhs, lhs_tiled, rhs_tiled, out)

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
