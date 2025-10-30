# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: Semaphore set/inc/wait operations generate correct D2M ops.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(2, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_sem_ops(lhs, rhs, out):
    @compute()
    async def comp(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
        sem: Semaphore,
    ):
        # Semaphore in signature to match datamovement arg count, but unused in compute.
        # Compute threads skip semaphore conversion (D2MToTTKernel.cpp:1300-1301).
        # TODO(#11): Addition required to create linalg.generic with tile ops.
        lhs_shard = lhs_cb.pop()
        rhs_shard = rhs_cb.pop()
        out_shard = out_cb.reserve()
        result = lhs_shard + rhs_shard
        out_shard.store(result)
        out_cb.pop()

    @datamovement()
    async def dm(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
        sem: Semaphore,
    ):
        cy = core_index(0)
        cx = core_index(1)

        if cx == 0:
            # Verify: sem.set() generates d2m.semaphore_set with multicast
            sem.set(1, core=(cy, 1), mcast=(1, 0))
            # Verify: sem.wait() generates d2m.semaphore_wait
            sem.wait(1, reset=0)
        else:
            # Verify: sem.inc() generates d2m.semaphore_inc
            sem.inc(1, core=(cy, 0))

    return Program(comp, dm)(lhs, rhs, out)


# CHECK-LABEL: func.func @test_sem_ops

# Verify: Semaphore operations in datamovement region
# CHECK: ^datamovement{{[0-9]+}}
# CHECK: d2m.semaphore_set
# CHECK: d2m.semaphore_wait
# CHECK: d2m.semaphore_inc

# CHECK-LOWERED-LABEL: func.func @test_sem_ops

# Verify: Lowered to ttmetal with kernel functions
# CHECK-LOWERED: ttmetal.enqueue_program
# CHECK-LOWERED: func.func private @datamovement_kernel{{[0-9]+}}
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}

lhs = torch.randn(64, 32)
rhs = torch.randn(64, 32)
out = torch.zeros(64, 32)
test_sem_ops(lhs, rhs, out)

# Note: Semaphore appears in compute signature to match datamovement arg count (required by d2m.generic),
# but is unused in compute code. Compute threads skip semaphore conversion (D2MToTTKernel.cpp:1300-1301).
# Using semaphores in compute regions causes: "failed to legalize unresolved materialization"
