# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: Explicit push/pop API for CircularBuffer
# Consumer: wait() + pop()
# Producer: reserve() + push()

import torch
from ttlang.d2m_api import *


@kernel(grid=(2, 2), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_cb_ops(lhs, rhs, out):
    @compute()
    def compute_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # Consumer pattern: wait() to acquire, pop() to release
        shard = lhs_cb.wait()
        shard2 = rhs_cb.wait()

        # Producer pattern: reserve() to acquire, push() to release
        out_shard = out_cb.reserve()
        result = shard + shard2
        out_shard.store(result)

        # Explicit releases
        lhs_cb.pop()  # Signal consumption of lhs
        rhs_cb.pop()  # Signal consumption of rhs
        out_cb.push()  # Signal production of out

    @datamovement()
    def dm_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    return Program(compute_thread, dm_thread)(lhs, rhs, out)


# CHECK: func.func @test_cb_ops

# Verify: CB operations in compute region
# CHECK: ^compute{{[0-9]+}}
# CHECK: %[[SHARD:.+]] = d2m.wait %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>
# CHECK: %[[OUT_SHARD:.+]] = d2m.reserve %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>
# CHECK: d2m.pop %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>
# CHECK: d2m.push %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>

# CHECK-LOWERED: func.func @test_cb_ops
# CHECK-LOWERED: emitc.call_opaque "cb_wait_front"
# CHECK-LOWERED: emitc.call_opaque "cb_pop_front"
# CHECK-LOWERED: emitc.call_opaque "cb_reserve_back"
# CHECK-LOWERED: emitc.call_opaque "cb_push_back"

lhs = torch.randn(64, 64)
rhs = torch.randn(64, 64)
out = torch.zeros(64, 64)
test_cb_ops(lhs, rhs, out)
