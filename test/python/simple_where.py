# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple where kernel - verifies where(cond, a, b) lowers to correct TTL ops and C++ code.

Tests single-tile ternary select: where(cond, true_val, false_val).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def where_kernel(cond, true_val, false_val, out):
    cond_dfb = ttl.make_dataflow_buffer_like(cond, shape=(1, 1), buffer_factor=2)
    true_dfb = ttl.make_dataflow_buffer_like(true_val, shape=(1, 1), buffer_factor=2)
    false_dfb = ttl.make_dataflow_buffer_like(false_val, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def where_compute():
        with cond_dfb.wait() as c, true_dfb.wait() as t, false_dfb.wait() as f, out_dfb.reserve() as out:
            out.store(ttl.math.where(c, t, f))

    @ttl.datamovement()
    def dm_read():
        with cond_dfb.reserve() as blk:
            tx = ttl.copy(cond[0, 0], blk)
            tx.wait()
        with true_dfb.reserve() as blk:
            tx = ttl.copy(true_val[0, 0], blk)
            tx.wait()
        with false_dfb.reserve() as blk:
            tx = ttl.copy(false_val[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# =============================================================================
# Initial IR Checks - Verify TTL dialect ops (compute kernel)
# =============================================================================

# CHECK-LABEL: func.func @where_compute
# CHECK-SAME: attributes {{{.*}}ttl.kernel_thread = #ttkernel.thread<compute>}

# CHECK-DAG: ttl.bind_cb{cb_index = 0
# CHECK-DAG: ttl.bind_cb{cb_index = 1
# CHECK-DAG: ttl.bind_cb{cb_index = 2
# CHECK-DAG: ttl.bind_cb{cb_index = 3

# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve

# CHECK: ttl.where

# CHECK: ttl.store

# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel
# =============================================================================

# CHECK-CPP: // where_compute
# CHECK-CPP: void kernel_main()

# CHECK-CPP: cb_wait_front(
# CHECK-CPP: cb_wait_front(
# CHECK-CPP: cb_wait_front(
# CHECK-CPP: cb_reserve_back(

# CHECK-CPP: tile_regs_acquire();

# CHECK-CPP: where_tile_init();
# CHECK-CPP: where_tile(

# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();

# CHECK-CPP: pack_tile<true>(

# CHECK-CPP: tile_regs_release();


device = ttnn.open_device(device_id=0)
cond = ttnn.from_torch(
    __import__("torch").randn(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
true_val = ttnn.from_torch(
    __import__("torch").randn(32, 32, dtype=__import__("torch").bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
false_val = ttnn.from_torch(
    __import__("torch").randn(32, 32, dtype=__import__("torch").bfloat16),
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
where_kernel(cond, true_val, false_val, out)
ttnn.close_device(device)
