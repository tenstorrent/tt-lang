# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test chained additions: (a + b) + c + d
# This verifies multiple operands work correctly

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)])
def test_chained_add(a, b, c, d, out):
    @compute()
    async def add_compute(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        d_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        a_tile = a_cb.pop()
        b_tile = b_cb.pop()
        c_tile = c_cb.pop()
        d_tile = d_cb.pop()
        out_tile = out_cb.reserve()

        # Chain: a + b + c + d
        result = a_tile + b_tile + c_tile + d_tile

        out_tile.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        d_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        pass

    return Program(add_compute, dm)(a, b, c, d, out)


# CHECK: func.func @test_chained_add

# Verify compute region has chained adds
# CHECK: ^compute{{[0-9]+}}
# CHECK: d2m.tile_add
# CHECK: d2m.tile_add
# CHECK: d2m.tile_add

# CHECK-LOWERED: func.func @test_chained_add
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

# Test: 1 + 2 + 3 + 4 = 10
a = torch.full((32, 32), 1.0)
b = torch.full((32, 32), 2.0)
c = torch.full((32, 32), 3.0)
d = torch.full((32, 32), 4.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"a: all 1.0")
print(f"b: all 2.0")
print(f"c: all 3.0")
print(f"d: all 4.0")
print(f"out: all -999.0")
print(f"Expected: all 10.0 (1+2+3+4)")

test_chained_add(a, b, c, d, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
# CHECK-OUTPUT: out[0, 0] = 10.0
print(f"out min/max/mean: {out.min().item():.1f} / {out.max().item():.1f} / {out.mean().item():.1f}")
# CHECK-OUTPUT: out min/max/mean: 10.0 / 10.0 / 10.0

expected = a + b + c + d
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected (1+2+3+4=10)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected all 10.0, got values from {out.min().item()} to {out.max().item()}")
