# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: TensorBlock.__matmul__ generates linalg.generic with matmul maps and tile_matmul.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_matmul(lhs, rhs, out):
    @compute()
    async def mm_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.pop()
        r = rhs_cb.pop()
        o = out_cb.reserve()
        result = l @ r
        o.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    return Program(mm_compute, dm)(lhs, rhs, out)


# CHECK: func.func @test_matmul

# Verify: compute region contains linalg.generic with matmul maps and tile_matmul
# CHECK: ^compute{{[0-9]+}}
# CHECK: %[[MM_RESULT:.+]] = linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
# CHECK: ^bb0(%[[IN0:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, %[[IN1:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, %[[ACC:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>):
# CHECK-NEXT: %[[TILE_MM:.+]] = "d2m.tile_matmul"(%[[IN0]], %[[IN1]], %[[ACC]])
# CHECK-NEXT: linalg.yield %[[TILE_MM]]

# CHECK-LOWERED: func.func @test_matmul
# CHECK-LOWERED: emitc.call_opaque "matmul_tiles"

lhs = torch.randn(32, 32)
rhs = torch.randn(32, 32)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"lhs[0:2, 0:2] =\n{lhs[0:2, 0:2]}")
print(f"rhs[0:2, 0:2] =\n{rhs[0:2, 0:2]}")
print(f"out[0:2, 0:2] =\n{out[0:2, 0:2]}")
expected = lhs @ rhs
print(f"expected[0:2, 0:2] =\n{expected[0:2, 0:2]}")

test_matmul(lhs, rhs, out)

print("\n=== AFTER KERNEL ===")
print(f"out[0:2, 0:2] =\n{out[0:2, 0:2]}")
print(f"expected[0:2, 0:2] =\n{expected[0:2, 0:2]}")
print(f"out min/max/mean: {out.min().item():.4f} / {out.max().item():.4f} / {out.mean().item():.4f}")
