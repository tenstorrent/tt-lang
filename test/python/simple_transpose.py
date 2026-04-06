# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple transpose kernel - verifies transpose lowers to correct TTL ops and C++ code.

Tests single-tile transpose.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def transpose_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def transpose_compute():
        with inp_dfb.wait() as inp, out_dfb.reserve() as out:
            out.store(ttl.math.transpose(inp))

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

# CHECK-LABEL: func.func @transpose_compute
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}

# CHECK: %[[IN_CB:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[OUT_CB:.+]] = ttl.bind_cb{cb_index = 1

# CHECK: ttl.cb_wait %[[IN_CB]]
# CHECK: ttl.cb_reserve %[[OUT_CB]]

# CHECK: ttl.transpose

# CHECK: ttl.store

# CHECK: ttl.cb_push %[[OUT_CB]]
# CHECK: ttl.cb_pop %[[IN_CB]]

# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel
# =============================================================================

# CHECK-CPP: // transpose_compute
# CHECK-CPP: void kernel_main()

# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(1),

# CHECK-CPP: tile_regs_acquire();

# CHECK-CPP: transpose_wh_init(
# CHECK-CPP: transpose_wh_tile(

# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();

# CHECK-CPP: pack_tile<true>(

# CHECK-CPP: tile_regs_release();

# CHECK-CPP: cb_push_back(get_compile_time_arg_val(1),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        inp = ttnn.from_torch(
            torch.randn(32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            torch.zeros(32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        transpose_kernel(inp, out)

    finally:
        ttnn.close_device(device)
