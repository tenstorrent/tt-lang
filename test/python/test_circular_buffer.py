# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: CircularBuffer.pop() generates d2m.wait and CircularBuffer.reserve() generates d2m.reserve.

import torch
from ttlang.d2m_api import *

@pykernel_gen(grid=(2, 2), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_cb_ops(lhs, rhs, out):
    @compute()
    async def compute_thread(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        # Verify: CB.pop() generates d2m.wait
        shard = lhs_cb.pop()
        shard2 = rhs_cb.pop()
        # Verify: CB.reserve() generates d2m.reserve
        out_shard = out_cb.reserve()
        result = shard + shard2
        out_shard.store(result)
        out_cb.pop()

    @datamovement()
    async def dm_thread(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        pass

    return Program(compute_thread, dm_thread)(lhs, rhs, out)

# CHECK: func.func @test_cb_ops

# Verify: CB operations in compute region
# CHECK: ^compute{{[0-9]+}}
# CHECK: %[[SHARD:.+]] = d2m.wait %{{.+}} : <tensor<1x1x!ttcore.tile<32x32, f32>>>
# CHECK: %[[OUT_SHARD:.+]] = d2m.reserve %{{.+}} : <tensor<1x1x!ttcore.tile<32x32, f32>>>

# CHECK-LOWERED: func.func @test_cb_ops
# CHECK-LOWERED: emitc.call_opaque "cb_wait_front"
# CHECK-LOWERED: emitc.call_opaque "cb_reserve_back"

lhs = torch.randn(64, 64)
rhs = torch.randn(64, 64)
out = torch.zeros(64, 64)
test_cb_ops(lhs, rhs, out)
