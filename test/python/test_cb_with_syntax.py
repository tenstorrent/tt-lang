# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: 'with' statement syntax for CircularBuffer reserve()
# Producer: with cb.reserve() as blk: ... (implicit push at end of scope)
# Consumer: wait() + pop() (explicit, unchanged)

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(2, 2), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_cb_with_syntax(lhs, rhs, out):
    @compute()
    def compute_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # Consumer pattern: wait() to acquire, pop() to release (explicit)
        shard = lhs_cb.wait()
        shard2 = rhs_cb.wait()

        # Producer pattern using 'with' statement: reserve() with implicit push()
        with out_cb.reserve() as out_shard:
            result = shard + shard2
            out_shard.store(result)
            # implicit out_cb.push() at end of scope

        # Explicit releases for consumer
        lhs_cb.pop()  # Signal consumption of lhs
        rhs_cb.pop()  # Signal consumption of rhs

    @datamovement()
    def dm_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    return Program(compute_thread, dm_thread)(lhs, rhs, out)


# CHECK: func.func @test_cb_with_syntax

# Verify: CB operations in compute region
# CHECK: ^compute{{[0-9]+}}
# CHECK: %[[SHARD:.+]] = d2m.wait %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>
# CHECK: %[[OUT_SHARD:.+]] = d2m.reserve %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>

# Verify: push is generated from the 'with' statement (implicit at end of scope)
# CHECK: d2m.push %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>

# Verify: pop operations for consumer
# CHECK: d2m.pop %{{.+}} : <tensor<1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>>>

# CHECK-LOWERED: func.func @test_cb_with_syntax
# CHECK-LOWERED: emitc.call_opaque "cb_wait_front"
# CHECK-LOWERED: emitc.call_opaque "cb_reserve_back"
# CHECK-LOWERED: emitc.call_opaque "cb_push_back"
# CHECK-LOWERED: emitc.call_opaque "cb_pop_front"

lhs = torch.randn(64, 64)
rhs = torch.randn(64, 64)
out = torch.zeros(64, 64)
test_cb_with_syntax(lhs, rhs, out)
