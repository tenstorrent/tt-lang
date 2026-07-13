# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Conditional reassignment lit test -- verifies that variables assigned
inside if/else branches produce scf.if with results and yields in the
initial MLIR (fix for ISSUE #380).

Tests datamovement if/else reassignment of a transfer handle (ttl.copy)
on a 2x1 grid.

NOTE: A compute if/else pattern (reassigning tensor block variables
across scf.if branches) is blocked by ISSUE #683
(convert-ttl-to-compute asserts when a value is stored from
multiple blocks). Re-enable once #683 is resolved.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(2, 1))
def cond_reassign_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    # Compute if/else reassignment is disabled until ISSUE #683 is fixed.
    # The AST tracer correctly generates scf.if with yields (verified
    # manually via TTLANG_INITIAL_MLIR), but convert-ttl-to-compute
    # asserts (isBeforeInBlock) when the scf.if result is stored.
    #
    # @ttl.compute()
    # def cond_compute():
    #     node_x, _ = ttl.node(dims=2)
    #     with inp_dfb.wait() as a, out_dfb.reserve() as o:
    #         result = a
    #         if node_x == 0:
    #             result = ttl.math.exp(a)
    #         else:
    #             result = ttl.math.tanh(a)
    #         o.store(result)

    @ttl.compute()
    def cond_compute():
        with inp_dfb.wait() as a, out_dfb.reserve() as o:
            o.store(a)

    # CHECK-LABEL: func.func @cond_dm_read
    @ttl.datamovement()
    def cond_dm_read():
        node_x, _ = ttl.node(dims=2)
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            if node_x == 1:
                tx = ttl.copy(inp[0, 1], blk)
            tx.wait()

    # The if-only pattern must synthesise an else that yields the
    # original transfer handle so the outer tx.wait() sees the right value.
    # CHECK: %[[TX0:.+]] = ttl.copy
    # CHECK: arith.cmpi eq
    # CHECK: %[[TX_RESULT:.+]] = scf.if
    # CHECK:   %[[TX1:.+]] = ttl.copy
    # CHECK:   scf.yield %[[TX1]]
    # CHECK: } else {
    # CHECK:   scf.yield %[[TX0]]
    # CHECK: }
    # CHECK: ttl.wait %[[TX_RESULT]]

    # CHECK-LABEL: func.func @cond_dm_write
    @ttl.datamovement()
    def cond_dm_write():
        node_x, _ = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            if node_x == 1:
                tx = ttl.copy(blk, out[0, 1])
            tx.wait()


device = ttnn.open_device(device_id=0)
try:
    lhs = ttnn.from_torch(
        __import__("torch").zeros(32, 64, dtype=__import__("torch").bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        __import__("torch").zeros(32, 64, dtype=__import__("torch").bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cond_reassign_kernel(lhs, out)
finally:
    ttnn.close_device(device)
