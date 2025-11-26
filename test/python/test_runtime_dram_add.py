# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Tilize-on-the-fly: DRAM scalar data is tilized inside compute kernel.
# 5 CBs: 2 scalar (DRAM inputs), 2 tiled (tilize destinations), 1 tiled (output).

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
def test_runtime_dram_add(lhs, rhs, lhs_tiled, rhs_tiled, out):
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")
    rhs_accessor = TensorAccessor(rhs, memory_space="DRAM")

    @compute()
    def add_compute(
        lhs_scalar_cb: CircularBuffer,  # CB 0: scalar from DRAM
        rhs_scalar_cb: CircularBuffer,  # CB 1: scalar from DRAM
        lhs_tiled_cb: CircularBuffer,   # CB 2: tilize destination
        rhs_tiled_cb: CircularBuffer,   # CB 3: tilize destination
        out_cb: CircularBuffer          # CB 4: tiled output
    ):
        # Wait for scalar data from DM threads
        l_scalar = lhs_scalar_cb.wait()
        r_scalar = rhs_scalar_cb.wait()

        # Tilize scalar -> tiled
        l_tiled = lhs_tiled_cb.reserve()
        r_tiled = rhs_tiled_cb.reserve()
        tilize(l_scalar, l_tiled)
        tilize(r_scalar, r_tiled)
        lhs_tiled_cb.push()
        rhs_tiled_cb.push()
        lhs_scalar_cb.pop()
        rhs_scalar_cb.pop()

        # Add tiled data
        l_tile = lhs_tiled_cb.wait()
        r_tile = rhs_tiled_cb.wait()
        o = out_cb.reserve()
        result = l_tile + r_tile
        o.store(result)
        out_cb.push()
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


# Verify: DRAM and L1 layouts defined
# CHECK: #layout = #ttcore.metal_layout<logical_shape = 32x32,
# CHECK-SAME: dram
# CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 32x32,
# CHECK-SAME: l1

# Verify: Globals use DRAM layout
# CHECK: ttcore.global @lhs = tensor<{{.*}}, #layout>
# CHECK: ttcore.global @rhs = tensor<{{.*}}, #layout>

# CHECK: func.func @test_runtime_dram_add

# Verify: Host tensors staged to DRAM via to_layout
# CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<32x32xf32> into tensor<{{.*}}, #layout>
# CHECK: d2m.stream_layout
# CHECK: d2m.to_layout %arg1, %{{.*}} : tensor<32x32xf32> into tensor<{{.*}}, #layout>

# Verify: Generic has 2 DRAM inputs, 2 L1 intermediates, 1 L1 output
# CHECK: d2m.generic
# CHECK: ins({{.*}} : tensor<{{.*}}, #layout>, tensor<{{.*}}, #layout>, tensor<{{.*}}, #layout1>, tensor<{{.*}}, #layout1>)
# CHECK: outs({{.*}} : tensor<{{.*}}, #layout1>)

# Verify: DM threads use get_global for DRAM source
# CHECK: ttcore.get_global @lhs : tensor<{{.*}}, #layout>
# CHECK: ttcore.get_global @rhs : tensor<{{.*}}, #layout>

# Verify: Compute has tilize and tile_add
# CHECK: ^compute{{[0-9]+}}
# CHECK: d2m.tile_tilize_block
# CHECK: linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel"]
# CHECK: "d2m.tile_add"

# Verify: Memory space attributes lowered
# CHECK-LOWERED: #dram = #ttcore.memory_space<dram>
# CHECK-LOWERED: #l1 = #ttcore.memory_space<l1>

# Verify: Globals lowered to DRAM memrefs
# CHECK-LOWERED: ttcore.global @lhs = memref<{{.*}}, #dram>
# CHECK-LOWERED: ttcore.global @rhs = memref<{{.*}}, #dram>

# CHECK-LOWERED: func.func @test_runtime_dram_add

# Verify: DRAM and L1 buffers created with addresses
# CHECK-LOWERED: "ttmetal.create_buffer"(){{.*}}#dram>
# CHECK-LOWERED: "ttmetal.create_buffer"(){{.*}}#l1>

# Verify: Compute kernel has tilize and add ops
# CHECK-LOWERED: emitc.call_opaque "experimental::tilize_block"
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

lhs = torch.full((32, 32), 2.0)
rhs = torch.full((32, 32), 3.0)
lhs_tiled = torch.zeros((32, 32))  # L1 intermediate
rhs_tiled = torch.zeros((32, 32))  # L1 intermediate
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
