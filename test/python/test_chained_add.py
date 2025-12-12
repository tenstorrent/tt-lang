# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Crashes in D2MInsertDstRegisterAccess: insufficient DST capacity (needs >4 slices for chained ops)
# XFAIL: *
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Test chained additions: (a + b) + c

import torch
from ttlang.d2m_api import *


@kernel(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1)])
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

        # Chain: (a + b) + c
        temp = a_tile + b_tile
        result = temp + c_tile

        out_tile.store(result)
        a_cb.pop()
        b_cb.pop()
        c_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_a(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        a_shard = a_cb.reserve()
        tx = dma(a_accessor[0, 0], a_shard)
        tx.wait()

    @datamovement()
    def dm_b(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        b_shard = b_cb.reserve()
        tx = dma(b_accessor[0, 0], b_shard)
        tx.wait()

    @datamovement()
    def dm_c(
        a_cb: CircularBuffer,
        b_cb: CircularBuffer,
        c_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        c_shard = c_cb.reserve()
        tx = dma(c_accessor[0, 0], c_shard)
        tx.wait()

    return Program(add_compute, dm_a, dm_b, dm_c)(a, b, c, out)


# CHECK: func.func @test_chained_add

# Verify compute region has chained adds
# CHECK: ^compute{{[0-9]+}}
# CHECK: d2m.tile_add
# CHECK: d2m.tile_add

# CHECK-LOWERED: func.func @test_chained_add
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

# Test: 1 + 2 + 3 = 6
a = torch.full((32, 32), 1.0)
b = torch.full((32, 32), 2.0)
c = torch.full((32, 32), 3.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"a: all 1.0")
print(f"b: all 2.0")
print(f"c: all 3.0")
print(f"out: all -999.0")
print(f"Expected: all 6.0 (1+2+3)")

test_chained_add(a, b, c, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
# CHECK-OUTPUT: out[0, 0] = 6.0
print(
    f"out min/max/mean: {out.min().item():.1f} / {out.max().item():.1f} / {out.mean().item():.1f}"
)
# CHECK-OUTPUT: out min/max/mean: 6.0 / 6.0 / 6.0

expected = a + b + c
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected (1+2+3=6)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(
        f"FAIL: Expected all 6.0, got values from {out.min().item()} to {out.max().item()}"
    )
