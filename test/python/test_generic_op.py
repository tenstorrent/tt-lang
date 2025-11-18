# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify d2m.generic creation with explicit grid and block_factors (empty indexing_maps/iterator_types).

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(2, 2), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_generic(lhs, rhs, out):
    @compute()
    async def comp(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.pop()
        r = rhs_cb.pop()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    return Program(comp, dm)(lhs, rhs, out)


# CHECK: func.func @test_generic(%[[ARG0:.+]]: tensor<{{[0-9]+}}x{{[0-9]+}}xf32>{{.*}}, %[[ARG1:.+]]: tensor<{{[0-9]+}}x{{[0-9]+}}xf32>{{.*}}, %[[ARG2:.+]]: tensor<{{[0-9]+}}x{{[0-9]+}}xf32>{{.*}})

# Verify: to_layout operations convert host tensors to device tensors
# CHECK: %[[DEV0:.+]] = d2m.to_layout %[[ARG0]]
# CHECK: %[[DEV1:.+]] = d2m.to_layout %[[ARG1]]

# Verify: d2m.generic with explicit datamovement form
# CHECK: %[[RESULT:.+]] = d2m.generic
# CHECK-SAME: block_factors = []
# CHECK-SAME: grid = #ttcore.grid<2x2>
# CHECK-SAME: indexing_maps = []
# CHECK-SAME: iterator_types = []
# CHECK-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
# CHECK-NEXT: ins(%[[DEV0]], %[[DEV1]]
# CHECK: outs({{%.*}} : tensor<2x2x1x1x!ttcore.tile

# CHECK: some string that does not exist!

# Verify: datamovement region
# CHECK: ^datamovement{{[0-9]+}}(%[[CB0:.+]]: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %[[CB1:.+]]: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %[[CB2:.+]]: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):

# Verify: compute region
# CHECK: ^compute{{[0-9]+}}(%[[CB0_C:.+]]: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %[[CB1_C:.+]]: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %[[CB2_C:.+]]: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):

# Verify: CB operations in compute region
# CHECK: %[[L:.+]] = d2m.wait %[[CB0_C]]
# CHECK: %[[R:.+]] = d2m.wait %[[CB1_C]]
# CHECK: %[[O:.+]] = d2m.reserve %[[CB2_C]]

# Verify: linalg.generic with tile_add inside compute region
# CHECK: %[[EMPTY:.+]] = d2m.empty()
# CHECK: %[[ADD_RESULT:.+]] = linalg.generic
# CHECK: ^bb0(%[[IN0:.+]]: !ttcore.tile<32x32, f32>, %[[IN1:.+]]: !ttcore.tile<32x32, f32>, %[[OUT:.+]]: !ttcore.tile<32x32, f32>):
# CHECK: %[[TILE_ADD:.+]] = "d2m.tile_add"(%[[IN0]], %[[IN1]])
# CHECK: linalg.yield %[[TILE_ADD]]

# CHECK: d2m.store %[[O]], %[[ADD_RESULT]]
# CHECK: d2m.wait %[[CB2_C]]

# Verify: to_layout converts result back to host and returns
# CHECK: %[[TO_HOST:.+]] = d2m.to_layout %[[RESULT]]
# CHECK: return %[[TO_HOST]]

# CHECK-LOWERED-LABEL: func.func @test_generic

# Verify: After pipeline, lowered to ttmetal ops
# CHECK-LOWERED: ttmetal.enqueue_program

# Verify: Kernel functions exist
# CHECK-LOWERED: func.func private @datamovement_kernel
# CHECK-LOWERED: func.func private @compute_kernel

# Verify: Core compute operations are present
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

lhs = torch.randn(64, 64)
rhs = torch.randn(64, 64)
out = torch.zeros(64, 64)
test_generic(lhs, rhs, out)
