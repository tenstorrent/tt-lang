# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Add kernel with explicit loop in compute - verifies for loops work inside kernels.

Uses a for loop to add the same values multiple times (accumulate pattern).
This tests loop support without requiring dynamic indices in data movement.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def add_loop_kernel(lhs, rhs, out):
    """Add kernel with loop in compute to accumulate results."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()

        # Initial: store l into output CB
        o = out_cb.reserve()
        o.store(l)
        out_cb.push()

        # Loop: read back, add r, store again (accumulate pattern)
        for i in range(4):
            accum = out_cb.wait()
            result = accum + r
            out_cb.pop()
            o = out_cb.reserve()
            o.store(result)
            out_cb.push()

        lhs_cb.pop()
        rhs_cb.pop()
        # Final value already pushed, DM will handle it

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_cb.reserve()
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_blk = rhs_cb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


# =============================================================================
# Initial IR Checks - Verify scf.for loop is generated in compute
# =============================================================================

# CHECK: #ttnn.buffer_type<l1>
# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>{{.*}}>

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# CB binding (alphabetical order of capture names: lhs_cb, out_cb, rhs_cb)
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1

# Initial: wait for inputs, reserve output, store, push
# CHECK: ttl.cb_wait %[[CB0]]
# CHECK: ttl.cb_wait %[[CB1]]
# CHECK: ttl.cb_reserve %[[CB2]]
# CHECK: ttl.cb_push %[[CB2]]

# For loop in compute (accumulate pattern)
# CHECK: scf.for
# CHECK: ttl.cb_wait %[[CB2]]
# CHECK: ttl.add
# CHECK: ttl.cb_pop %[[CB2]]
# CHECK: ttl.cb_reserve %[[CB2]]
# CHECK: ttl.cb_push %[[CB2]]

# Finalize: pop inputs
# CHECK: ttl.cb_pop %[[CB0]]
# CHECK: ttl.cb_pop %[[CB1]]

# =============================================================================
# C++ Kernel Checks - Verify for loop in generated compute code
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()

# Initial: wait for inputs, reserve and push output
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),

# For loop with accumulate pattern
# CHECK-CPP: for (size_t {{.*}} < {{.*}};
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(2),
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(2),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),

# Finalize: pop inputs
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Loop Add Kernel Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rhs = ttnn.from_torch(
            rhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        rhs = ttnn.to_memory_config(rhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling loop add kernel...")
        add_loop_kernel(lhs, rhs, out)

        print("=== Loop Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
