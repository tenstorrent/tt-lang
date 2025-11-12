# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Verify: Runtime execution of add operation works correctly on hardware.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_runtime_add(lhs, rhs, out):
    lhs_stream = Stream(lhs)
    rhs_stream = Stream(rhs)

    @compute()
    async def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.pop()
        r = rhs_cb.pop()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        out_cb.pop()

    @datamovement()
    async def dm_lhs(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_stream[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    async def dm_rhs(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_stream[0, 0], rhs_shard)
        tx.wait()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


# CHECK: func.func @test_runtime_add

# Verify: compute region contains linalg.generic with identity maps and tile_add
# CHECK: ^compute{{[0-9]+}}
# CHECK: %[[ADD_RESULT:.+]] = linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel"]
# CHECK: ^bb0(%[[IN0:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, %[[IN1:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, %[[OUT:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>):
# CHECK-NEXT: %[[TILE_ADD:.+]] = "d2m.tile_add"(%[[IN0]], %[[IN1]])
# CHECK-NEXT: linalg.yield %[[TILE_ADD]]

# CHECK-LOWERED: func.func @test_runtime_add
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

test_runtime_add(lhs, rhs, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
# CHECK-OUTPUT: out[0, 0] = 5.0
print(
    f"out min/max/mean: {out.min().item():.1f} / {out.max().item():.1f} / {out.mean().item():.1f}"
)
# CHECK-OUTPUT: out min/max/mean: 5.0 / 5.0 / 5.0

expected = lhs + rhs
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected (2.0 + 3.0 = 5.0)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(
        f"FAIL: Expected all 5.0, got values from {out.min().item()} to {out.max().item()}"
    )
