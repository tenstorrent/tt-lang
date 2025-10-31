# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify: Stream() creates d2m.stream_layout ops with storage and view layouts.

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_stream(lhs, rhs, out):
    lhs_stream = Stream(lhs)

    @compute()
    async def comp(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # TODO(#11): Addition required to create linalg.generic with tile ops.
        # Can simplify to just CB operations once compiler handles regions without tile ops.
        lhs_shard = lhs_cb.pop()
        rhs_shard = rhs_cb.pop()
        out_shard = out_cb.reserve()
        result = lhs_shard + rhs_shard
        out_shard.store(result)
        out_cb.pop()

    @datamovement()
    async def dm0(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        shard = lhs_cb.reserve()
        # Verify: Stream is accessed via indexing
        tx = dma(lhs_stream[0, 0], shard)
        tx.wait()

    @datamovement()
    async def dm1(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        pass

    return Program(comp, dm0, dm1)(lhs, rhs, out)


# CHECK-DAG: #[[STORAGE_LAYOUT:.+]] = #ttcore.metal_layout<{{.*}}, l1>
# CHECK-DAG: #[[VIEW_LAYOUT:.+]] = #ttcore.metal_layout<{{.*}}, l1, index_map =

# CHECK-LABEL: func.func @test_stream

# Verify: First argument marked as stream
# CHECK-SAME: (%[[ARG0:.+]]: tensor<1x1x1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, #[[STORAGE_LAYOUT]]> {d2m.stream = true}

# Verify: Storage created with layout WITHOUT index_map (becomes ShardLayoutAttr after bufferization)
# CHECK: %[[STORAGE:.+]] = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, #[[STORAGE_LAYOUT]]>

# Verify: stream_layout connects input to storage, returns view WITH index_map (becomes ViewLayoutAttr)
# CHECK: %[[STREAM:.+]] = "d2m.stream_layout"(%[[ARG0]], %[[STORAGE]])
# CHECK-SAME: -> tensor<1x1x1x1x!ttcore.tile<{{[0-9]+}}x{{[0-9]+}}, {{.*}}>, #[[VIEW_LAYOUT]]>

# Verify: Stream view used as input to d2m.generic (low-level DSL uses empty indexing_maps/iterator_types)
# Note: Low-level DSL provides explicit grid/block_factors and manual thread logic.
# CHECK: d2m.generic {block_factors = [1, 1, 1, 1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = []
# CHECK-NEXT: ins(%[[STREAM]]

# CHECK-LOWERED-LABEL: func.func @test_stream

# Verify: Lowered to ttmetal with kernel functions
# CHECK-LOWERED: ttmetal.enqueue_program
# CHECK-LOWERED: func.func private @datamovement_kernel{{[0-9]+}}
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}

lhs = torch.randn(32, 32)
rhs = torch.randn(32, 32)
out = torch.zeros(32, 32)
test_stream(lhs, rhs, out)
