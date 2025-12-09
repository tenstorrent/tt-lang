# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Verify: Runtime execution of exp operation works correctly on hardware.

import torch
from ttlang.d2m_api import *
from ttlang.operators import exp


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_runtime_exp(input, out):
    input_accessor = TensorAccessor(input)

    @compute()
    async def exp_compute(input_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        o = out_cb.reserve()
        result = exp(inp)
        o.store(result)
        input_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_input(input_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()

    return Program(exp_compute, dm_input)(input, out)


# CHECK: func.func @test_runtime_exp

# Verify: compute region contains linalg.generic with identity maps and tile_exp
# CHECK: ^compute{{[0-9]+}}
# CHECK: %[[EXP_RESULT:.+]] = linalg.generic
# CHECK-SAME: iterator_types = ["parallel", "parallel"]
# CHECK: ^bb0(%[[IN0:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, %[[OUT:.+]]: !ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>):
# CHECK-NEXT: %[[TILE_EXP:.+]] = "d2m.tile_exp"(%[[IN0]])
# CHECK-NEXT: linalg.yield %[[TILE_EXP]]

# CHECK-LOWERED: func.func @test_runtime_exp
# CHECK-LOWERED: emitc.call_opaque "exp_tile"

# Use simple known values for testing: exp(1.0) ≈ 2.718
input = torch.full((32, 32), 1.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"input: all 1.0")
print(f"out: all -999.0")
print(f"Expected: all ~2.718")

test_runtime_exp(input, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item():.3f}")
print(
    f"out min/max/mean: {out.min().item():.3f} / {out.max().item():.3f} / {out.mean().item():.3f}"
)
# CHECK-OUTPUT: out min/max/mean: 2.7{{[0-9]+}} / 2.7{{[0-9]+}} / 2.7{{[0-9]+}}

expected = torch.exp(input)
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected exp(1.0) ≈ 2.718")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(f"FAIL: Expected {expected.mean().item():.3f}, got {out.mean().item():.3f}")
