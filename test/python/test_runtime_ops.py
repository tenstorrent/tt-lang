# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s
# RUN: FileCheck %s --check-prefix=CHECK-LOWERED < %t.final.mlir

# Verify that runtime operations (enqueue_write_buffer, enqueue_read_buffer, tilize, untilize)
# are generated correctly in the lowered MLIR. These ops are critical for data transfer
# between host and device.

import torch
from ttlang.d2m_api import *


@kernel(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def test_runtime_ops(lhs, rhs, out):
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)

    @compute()
    def add_compute(
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

    @datamovement()
    def dm_lhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()

    @datamovement()
    def dm_rhs(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, out)


# CHECK-LOWERED-LABEL: func.func @test_runtime_ops

# Verify: Input 0 - create buffers and write to device
# CHECK-LOWERED: %[[BUF0:.+]] = "ttmetal.create_buffer"()
# CHECK-LOWERED: %[[BUF1:.+]] = "ttmetal.create_buffer"()
# CHECK-LOWERED: "ttmetal.enqueue_write_buffer"(%arg0, %[[BUF1]])

# Verify: Input 0 - enqueue tilize program
# CHECK-LOWERED: "ttmetal.enqueue_program"({{.*}})

# Verify: Input 1 - create buffers and write to device
# CHECK-LOWERED: %{{.+}} = "ttmetal.create_buffer"()
# CHECK-LOWERED: %{{.+}} = "ttmetal.create_buffer"()
# CHECK-LOWERED: %[[BUF4:.+]] = "ttmetal.create_buffer"()
# CHECK-LOWERED: "ttmetal.enqueue_write_buffer"(%arg1, %[[BUF4]])

# Verify: Input 1 - enqueue tilize program
# CHECK-LOWERED: "ttmetal.enqueue_program"({{.*}})

# Verify: Main compute program enqueued
# CHECK-LOWERED: "ttmetal.enqueue_program"({{.*}})

# Verify: Untilize program enqueued
# CHECK-LOWERED: "ttmetal.enqueue_program"({{.*}})

# Verify: Read result back to host
# CHECK-LOWERED: "ttmetal.enqueue_read_buffer"({{.+}}, %arg2)

# Verify: Cleanup and return
# CHECK-LOWERED: "ttmetal.finish"()
# CHECK-LOWERED: return %arg2

# Verify kernel functions exist with tilize operations
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}
# CHECK-LOWERED: emitc.call_opaque "tilize_init"
# CHECK-LOWERED: emitc.call_opaque "experimental::tilize_block"

# Verify second tilize kernel
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}
# CHECK-LOWERED: emitc.call_opaque "tilize_init"
# CHECK-LOWERED: emitc.call_opaque "experimental::tilize_block"

# Verify main compute kernel with add
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}
# CHECK-LOWERED: emitc.call_opaque "add_binary_tile"

# Verify untilize kernel
# CHECK-LOWERED: func.func private @compute_kernel{{[0-9]+}}
# CHECK-LOWERED: emitc.call_opaque "untilize_init"
# CHECK-LOWERED: emitc.call_opaque "experimental::untilize_block"

lhs = torch.randn(32, 32)
rhs = torch.randn(32, 32)
out = torch.randn(32, 32)

test_runtime_ops(lhs, rhs, out)
