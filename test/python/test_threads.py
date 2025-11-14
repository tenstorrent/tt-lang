# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: @compute() and @datamovement() decorators create threads with correct ThreadAttr.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(2, 2), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_thread_types(lhs, rhs, out):
    @datamovement()
    async def dm_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    @compute()
    async def compute_thread(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    return Program(compute_thread, dm_thread)(lhs, rhs, out)


# CHECK: func.func @test_thread_types

# Verify: threads attribute contains both thread types
# CHECK: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]

# Verify: datamovement region exists
# CHECK: ^datamovement{{[0-9]+}}

# Verify: compute region exists
# CHECK: ^compute{{[0-9]+}}

# CHECK-LOWERED-LABEL: func.func @test_thread_types

# Verify: After pipeline, lowered to ttmetal ops
# CHECK-LOWERED: ttmetal.enqueue_program

# Verify: Kernel functions exist with correct thread types
# CHECK-LOWERED: func.func private @datamovement_kernel{{[0-9]+}}{{.*}}ttkernel.thread = #ttkernel.thread<noc>
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}{{.*}}ttkernel.thread = #ttkernel.thread<compute>

lhs = torch.randn(64, 64)
rhs = torch.randn(64, 64)
out = torch.zeros(64, 64)
test_thread_types(lhs, rhs, out)
