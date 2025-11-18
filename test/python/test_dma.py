# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: dma() generates d2m.dma and MemTx.wait() generates d2m.dma_wait.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_dma_ops(lhs, rhs, out):
    lhs_accessor = TensorAccessor(lhs)

    @compute()
    def comp(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # TODO(#11): Addition required to create linalg.generic with tile ops.
        # Can simplify to just CB operations once compiler handles regions without tile ops.
        lhs_shard = lhs_cb.wait()
        rhs_shard = rhs_cb.wait()
        out_shard = out_cb.reserve()
        result = lhs_shard + rhs_shard
        out_shard.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    def dm0(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        shard = lhs_cb.reserve()
        # Verify: dma() generates d2m.dma and returns MemTx
        tx = dma(lhs_accessor[0, 0], shard)
        # Verify: MemTx.wait() generates d2m.dma_wait
        tx.wait()

    @datamovement()
    def dm1(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    return Program(comp, dm0, dm1)(lhs, rhs, out)


# CHECK-LABEL: func.func @test_dma_ops

# Verify: DMA operations in datamovement region
# CHECK: ^datamovement{{[0-9]+}}
# CHECK: %[[SHARD:.+]] = d2m.reserve
# CHECK: %[[TX:.+]] = d2m.dma %{{.+}}, %[[SHARD]]
# CHECK: d2m.dma_wait %[[TX]]

# CHECK-LOWERED-LABEL: func.func @test_dma_ops

# Verify: Lowered to ttmetal with kernel functions
# CHECK-LOWERED: ttmetal.enqueue_program
# CHECK-LOWERED: func.func private @datamovement_kernel{{[0-9]+}}
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}

lhs = torch.randn(32, 32)
rhs = torch.randn(32, 32)
out = torch.zeros(32, 32)
test_dma_ops(lhs, rhs, out)
