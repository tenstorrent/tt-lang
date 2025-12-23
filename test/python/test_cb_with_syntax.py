# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: 'with' statement syntax for CircularBuffer
# Acquire ops left-to-right, release ops in reverse order

import torch
from ttlang.ttl_api import *


@pykernel_gen(grid=(2, 2), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_cb_with_syntax(lhs, rhs, out):
    @compute()
    def compute_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        with (
            lhs_cb.wait() as shard,
            rhs_cb.wait() as shard2,
            out_cb.reserve() as out_shard,
        ):
            result = shard + shard2
            out_shard.store(result)

    @datamovement()
    def dm_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    return Program(compute_thread, dm_thread)(lhs, rhs, out)


# CHECK: func.func @test_cb_with_syntax
# CHECK: ^compute{{[0-9]+}}(%[[CB0:.+]]: !d2m.cb<{{.*}}>, %[[CB1:.+]]: !d2m.cb<{{.*}}>, %[[CB2:.+]]: !d2m.cb<{{.*}}>):

# Acquire ops in left-to-right order: wait(cb0), wait(cb1), reserve(cb2)
# CHECK: d2m.wait %[[CB0]]
# CHECK: d2m.wait %[[CB1]]
# CHECK: d2m.reserve %[[CB2]]

# Release ops in reverse order: push(cb2), pop(cb1), pop(cb0)
# CHECK: d2m.push %[[CB2]]
# CHECK: d2m.pop %[[CB1]]
# CHECK: d2m.pop %[[CB0]]

# CHECK-LOWERED: func.func @test_cb_with_syntax
# CHECK-LOWERED: emitc.call_opaque "cb_wait_front"
# CHECK-LOWERED: emitc.call_opaque "cb_wait_front"
# CHECK-LOWERED: emitc.call_opaque "cb_reserve_back"
# CHECK-LOWERED: emitc.call_opaque "cb_push_back"
# CHECK-LOWERED: emitc.call_opaque "cb_pop_front"
# CHECK-LOWERED: emitc.call_opaque "cb_pop_front"

lhs = torch.randn(64, 64)
rhs = torch.randn(64, 64)
out = torch.zeros(64, 64)
test_cb_with_syntax(lhs, rhs, out)
