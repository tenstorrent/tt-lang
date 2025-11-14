# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# UNSUPPORTED: system-darwin
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

import torch
from ttlang.d2m_api import *
from ttlang.operators import transpose


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def test_runtime_transpose(input, out):
    input_accessor = TensorAccessor(input)

    @compute()
    async def transpose_compute(input_cb: CircularBuffer, out_cb: CircularBuffer):
        inp = input_cb.wait()
        o = out_cb.reserve()
        result = transpose(inp)
        o.store(result)
        input_cb.pop()
        out_cb.push()

    @datamovement()
    async def dm_input(input_cb: CircularBuffer, out_cb: CircularBuffer):
        input_shard = input_cb.reserve()
        tx = dma(input_accessor[0, 0], input_shard)
        tx.wait()

    return Program(transpose_compute, dm_input)(input, out)


# CHECK: func.func @test_runtime_transpose
# CHECK: "d2m.tile_transpose"

input = torch.randn((32, 32))
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Testing transpose compilation")

test_runtime_transpose(input, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print("PASS: transpose compiled successfully")
# CHECK-OUTPUT: PASS: transpose compiled successfully
