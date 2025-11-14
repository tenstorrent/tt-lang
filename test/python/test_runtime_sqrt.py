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
from ttlang.operators import sqrt


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_runtime_sqrt(input, out):
    input_accessor = TensorAccessor(input)

    @compute()
    async def sqrt_compute(input_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        o = out_cb.reserve()
        result = sqrt(inp)
        o.store(result)
        input_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_input(input_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()

    return Program(sqrt_compute, dm_input)(input, out)


# CHECK: func.func @test_runtime_sqrt
# CHECK: ^compute{{[0-9]+}}
# CHECK: %{{.+}} = linalg.generic
# CHECK: "d2m.tile_sqrt"

# CHECK-LOWERED: func.func @test_runtime_sqrt
# CHECK-LOWERED: emitc.call_opaque "sqrt_tile"

input = torch.full((32, 32), 4.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"input: all 4.0")
print(f"Expected: all 2.0")

test_runtime_sqrt(input, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.3f}")
# CHECK-OUTPUT: out[0, 0] = 2.0{{[0-9]*}}

expected = torch.sqrt(input)
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected sqrt(4.0) = 2.0")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected {expected.mean().item():.3f}, got {out.mean().item():.3f}")
