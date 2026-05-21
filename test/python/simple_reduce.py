# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple reduce kernel - verifies reduce_sum lowers to correct TTL ops and C++ code.

Tests single-tile scalar reduction (dims=[0,1]).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def reduce_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def reduce_compute():
        with inp_dfb.wait() as inp, out_dfb.reserve() as out:
            out.store(ttl.math.reduce_sum(inp, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# Initial IR Checks - Verify TTL dialect ops (compute kernel)
# =============================================================================

# CHECK-LABEL: func.func @reduce_compute
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}

# CHECK: ttl.bind_cb{cb_index = 0
# CHECK: ttl.bind_cb{cb_index =

# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve

# The DSL synthesizes a fill(1.0) tile internally as the LLK scaler operand
# (the user no longer supplies a scaler; the unit reduce passes c=1 to the LLK).
# CHECK: ttl.fill 1.000000e+00
# CHECK: ttl.reduce

# CHECK: ttl.store

# CHECK: ttl.cb_push
# CHECK: ttl.cb_pop

# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel
# =============================================================================

# CHECK-CPP: // reduce_compute
# CHECK-CPP: void kernel_main()
# CHECK-CPP-DAG: CircularBuffer [[CB0:.*]](get_compile_time_arg_val(0));
# CHECK-CPP-DAG: CircularBuffer [[CB1:.*]](get_compile_time_arg_val(1));
# CHECK-CPP-DAG: CircularBuffer [[CB2:.*]](get_compile_time_arg_val(2));

# Scaler tile materialized inside compute via fill_tile, then reduce_init / reduce_tile.
# CHECK-CPP: fill_tile_init();
# CHECK-CPP: fill_tile(
# CHECK-CPP: reduce_init<
# CHECK-CPP: reduce_tile<
# CHECK-CPP: pack_tile<true>(


device = ttnn.open_device(device_id=0)
inp = ttnn.from_torch(
    __import__("torch").ones(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
out = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
reduce_kernel(inp, out)
ttnn.close_device(device)
