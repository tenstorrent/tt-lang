# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *
from ttlang.operators import maximum


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_runtime_maximum(lhs, rhs, out):
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)

    @compute()
    async def maximum_compute(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = maximum(l, r)
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_lhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    async def dm_rhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()

    return Program(maximum_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


# CHECK: func.func @test_runtime_maximum
# CHECK: ^compute{{[0-9]+}}
# CHECK: %{{.+}} = linalg.generic
# CHECK: "d2m.tile_maximum"

# CHECK-LOWERED: func.func @test_runtime_maximum

lhs = torch.full((32, 32), 2.0)
rhs = torch.full((32, 32), 3.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"lhs: all 2.0, rhs: all 3.0")
print(f"Expected: all 3.0")

test_runtime_maximum(lhs, rhs, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.3f}")

expected = torch.maximum(lhs, rhs)
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected maximum(2.0, 3.0) = 3.0")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected {expected.mean().item():.3f}, got {out.mean().item():.3f}")
