# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Tests liveness-based DST allocation with chained operations
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test chained additions: (a + b) + c

import torch
from ttlang.d2m_api import *
from ttlang.operators import add_into


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
def test_chained_add(a, b, c, out):
    a_accessor = TensorAccessor(a)
    b_accessor = TensorAccessor(b)
    c_accessor = TensorAccessor(c)

    @compute()
    def add_compute(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        a_tile = a_cb.wait()
        b_tile = b_cb.wait()
        c_tile = c_cb.wait()
        out_tile = out_cb.reserve()

        # Chained: (a + b) + c - directly without intermediate storage
        temp = a_tile + b_tile
        result = temp + c_tile

        out_tile.store(result)
        a_cb.pop()
        b_cb.pop()
        c_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_inputs(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        a_shard = a_cb.reserve()
        tx = dma(a_accessor[0, 0], a_shard)
        tx.wait()
        a_cb.push()

        b_shard = b_cb.reserve()
        tx = dma(b_accessor[0, 0], b_shard)
        tx.wait()
        b_cb.push()

        c_shard = c_cb.reserve()
        tx = dma(c_accessor[0, 0], c_shard)
        tx.wait()
        c_cb.push()

    return Program(add_compute, dm_inputs)(a, b, c, out)


# CHECK: func.func @test_chained_add

# Verify compute region has add
# CHECK: ^compute{{[0-9]+}}
# CHECK: d2m.tile_add

# CHECK-LOWERED: func.func @test_chained_add
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

# Test: (10 + 20) + 100 = 130 using add_into for second add
a = torch.full((32, 32), 10.0)
b = torch.full((32, 32), 20.0)
c = torch.full((32, 32), 100.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"a: all {a[0,0].item()}")
print(f"b: all {b[0,0].item()}")
print(f"c: all {c[0,0].item()}")
print(f"out: all {out[0,0].item()}")
print(f"Expected: out=130.0 ((10+20)+100)")

test_chained_add(a, b, c, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print("out:")
print(out)

expected = (a + b) + c
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("\nPASS: Output matches expected ((10+20)+100=130)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"\nFAIL: Expected 130.0, got range [{out.min().item():.1f}, {out.max().item():.1f}]")
