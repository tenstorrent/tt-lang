# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: TensorBlock.__add__ generates linalg.generic with identity maps and tile_add.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_add(lhs, rhs, out):
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
    def dm(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        pass

    return Program(add_compute, dm)(lhs, rhs, out)


# CHECK: func.func @test_add

# Verify: compute region contains linalg.generic with identity maps and tile_add
# CHECK: ^compute{{[0-9]+}}
# CHECK: %[[ADD_RESULT:.+]] = linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel"]
# CHECK: ^bb0(%[[IN0:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, %[[IN1:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, %[[OUT:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>):
# CHECK-NEXT: %[[TILE_ADD:.+]] = "d2m.tile_add"(%[[IN0]], %[[IN1]])
# CHECK-NEXT: linalg.yield %[[TILE_ADD]]

# CHECK-LOWERED: func.func @test_add
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

lhs = torch.randn(32, 32)
rhs = torch.randn(32, 32)
out = torch.randn(32, 32)

test_add(lhs, rhs, out)
