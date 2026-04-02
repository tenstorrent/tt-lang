# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple fill kernel - verifies fill(1.0) lowers to correct TTL ops and C++ code.

Tests single-tile fill with constant f32 value.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def fill_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fill_compute():
        with inp_dfb.wait() as _in, out_dfb.reserve() as out:
            out.store(ttl.math.fill(out, 1.0))

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

# CHECK-LABEL: func.func @fill_compute
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}

# CHECK: ttl.bind_cb{cb_index = 0
# CHECK: %[[OUT_CB:.+]] = ttl.bind_cb{cb_index = 1

# CHECK: ttl.cb_reserve %[[OUT_CB]]

# CHECK: ttl.fill

# CHECK: ttl.store

# CHECK: ttl.cb_push %[[OUT_CB]]

# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel
# =============================================================================

# CHECK-CPP: // fill_compute
# CHECK-CPP: void kernel_main()

# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(1),

# CHECK-CPP: tile_regs_acquire();

# CHECK-CPP: fill_tile_init();
# CHECK-CPP: fill_tile(

# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();

# CHECK-CPP: pack_tile<true>(

# CHECK-CPP: tile_regs_release();

# CHECK-CPP: cb_push_back(get_compile_time_arg_val(1),


device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
b = ttnn.from_torch(
    __import__("torch").zeros(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
fill_kernel(a, b)
ttnn.close_device(device)
