# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Crashes in D2MInsertDstRegisterAccess because no dst register operations (zero compute ops)
# XFAIL: *
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Simplest possible test: write a constant value to output

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_write_constant(inp, out):
    inp_accessor = TensorAccessor(inp)

    @compute()
    async def write_compute(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        inp_tile = inp_cb.pop()
        out_tile = out_cb.reserve()
        out_tile.store(inp_tile)
        out_cb.pop()

    @datamovement()
    async def dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        inp_shard = inp_cb.reserve()
        tx = dma(inp_accessor[0, 0], inp_shard)
        tx.wait()

    return Program(write_compute, dm)(inp, out)


# CHECK: func.func @test_write_constant

# Verify compute region stores the input tile
# CHECK: ^compute{{[0-9]+}}
# CHECK: d2m.store

# CHECK-LOWERED: func.func @test_write_constant

# Test: Input is all 42.0, output should also be all 42.0
inp = torch.full((32, 32), 42.0)
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"inp: all 42.0")
print(f"out: all -999.0")

test_write_constant(inp, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
# CHECK-OUTPUT: out[0, 0] = 42.0
print(f"out min/max: {out.min().item():.1f} / {out.max().item():.1f}")
# CHECK-OUTPUT: out min/max: 42.0 / 42.0

if torch.allclose(out, inp, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches input (constant 42.0)")
    # CHECK-OUTPUT: PASS: Output matches input
else:
    print(
        f"FAIL: Expected all 42.0, got values from {out.min().item()} to {out.max().item()}"
    )
