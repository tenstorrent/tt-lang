# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Reduce -> broadcast with nested with-stmts.

Verifies that scalar reduce_sum feeding a ROW broadcast through nested
with-stmt scopes (reserve/wait on an intermediate DFB) compiles correctly.
The compiler must trace through the CB push/wait cycle across nested blocks
to detect the reduce and select the correct hardware broadcast type (SCALAR
instead of the frontend's ROW, since REDUCE_SCALAR places its result at tile
element [0,0], not across the full row).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def reduce_bcast_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    red_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, sc_dfb.wait() as scaler_blk:
            with red_dfb.reserve() as red_blk:
                red_blk.store(ttl.math.reduce_sum(inp_blk, scaler_blk, dims=[0, 1]))
            with red_dfb.wait() as red_blk, out_dfb.reserve() as out_blk:
                out_blk.store(ttl.math.broadcast(red_blk, out_blk, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0:2, 0:2], blk).wait()
        with sc_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0:2, 0]).wait()


# =============================================================================
# Initial IR Checks - Verify nested with-stmts produce reduce → CB → broadcast
# =============================================================================

# CHECK-LABEL: func.func @compute_fn
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}

# Reduce stores into intermediate CB, then broadcast reads from it.
# CHECK: ttl.reduce
# CHECK: ttl.store
# CHECK: ttl.cb_push
# CHECK: ttl.cb_wait
# CHECK: ttl.bcast

# =============================================================================
# C++ Kernel Checks - Verify hardware bcast type is SCALAR and
# reduce_init is present.
# =============================================================================

# CHECK-CPP: // compute_fn
# CHECK-CPP: void kernel_main()

# Reduce init and tile operation.
# CHECK-CPP: reduce_init<
# CHECK-CPP: reduce_tile<

# Broadcast must use BroadcastType::SCALAR (the frontend sets ROW based on
# dims alone, but the lowering selects SCALAR because REDUCE_SCALAR places
# its result at [0,0], not across row 0).
# CHECK-CPP: unary_bcast_init<BroadcastType::SCALAR>
# CHECK-CPP: unary_bcast<BroadcastType::SCALAR>


device = ttnn.open_device(device_id=0)
inp = ttnn.from_torch(
    __import__("torch").ones(64, 64, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
scaler = ttnn.from_torch(
    __import__("torch").ones(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
out = ttnn.from_torch(
    __import__("torch").zeros(64, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
reduce_bcast_kernel(inp, scaler, out)
ttnn.close_device(device)
