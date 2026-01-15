# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Add kernel using 'with' pattern for CB lifecycle management.

The 'with' statement automatically handles:
- Acquire: wait/reserve at context entry
- Release: pop/push at context exit (in reverse order)
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def add_with_kernel(lhs, rhs, out):
    """Add kernel using 'with' pattern for automatic CB lifecycle."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        # 'with' handles wait/reserve at entry, pop/push at exit
        with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
            result = l + r
            o.store(result)
        # Automatic: out_cb.push(), rhs_cb.pop(), lhs_cb.pop() (reverse order)

    @ttl.datamovement()
    def dm_read():
        # 'with' for reserve/push pattern
        with lhs_cb.reserve() as lhs_blk:
            tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
            tx_lhs.wait()
        # Automatic: lhs_cb.push()

        with rhs_cb.reserve() as rhs_blk:
            tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
            tx_rhs.wait()
        # Automatic: rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        # 'with' for wait/pop pattern
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()
        # Automatic: out_cb.pop()

    return ttl.Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - Verify 'with' generates correct CB ops
# =============================================================================

# CHECK: #ttnn.buffer_type<l1>
# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>{{.*}}>

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# CB binding (alphabetical order: lhs_cb=0, out_cb=2, rhs_cb=1)
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1

# 'with' entry: wait for inputs, reserve output (with CB association)
# CHECK: %[[L:.+]] = ttl.cb_wait %[[CB0]]
# CHECK: ttl.attach_cb %[[L]], %[[CB0]]
# CHECK: %[[R:.+]] = ttl.cb_wait %[[CB1]]
# CHECK: ttl.attach_cb %[[R]], %[[CB1]]
# CHECK: %[[O:.+]] = ttl.cb_reserve %[[CB2]]
# CHECK: ttl.attach_cb %[[O]], %[[CB2]]

# Add operation
# CHECK: ttl.add

# store() attaches result to output CB
# CHECK: ttl.attach_cb %{{.+}}, %[[CB2]]

# 'with' exit: push output, pop inputs (reverse order)
# CHECK: ttl.cb_push %[[CB2]]
# CHECK: ttl.cb_pop %[[CB1]]
# CHECK: ttl.cb_pop %[[CB0]]

# =============================================================================
# Initial IR Checks - Data movement with 'with' pattern
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0 : i32, 1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>}

# First CB: reserve (with CB association), copy, push
# CHECK: ttl.cb_reserve
# CHECK: ttl.attach_cb
# CHECK: ttl.copy {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait
# CHECK: ttl.cb_push

# Second CB: reserve (with CB association), copy, push
# CHECK: ttl.cb_reserve
# CHECK: ttl.attach_cb
# CHECK: ttl.copy {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait
# CHECK: ttl.cb_push

# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [2 : i32], ttl.kernel_thread = #ttkernel.thread<noc>}

# Output CB: wait (with CB association), copy, pop
# CHECK: ttl.cb_wait
# CHECK: ttl.attach_cb
# CHECK: ttl.copy {{.*}} -> !ttl.transfer_handle<write>
# CHECK: ttl.wait
# CHECK: ttl.cb_pop

# =============================================================================
# C++ Kernel Checks - Verify generated code
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()

# Wait for inputs
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),

# Reserve output
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),

# Add operation
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();
# CHECK-CPP: pack_tile<false>(
# CHECK-CPP: tile_regs_release();

# Push output, pop inputs (reverse order from 'with' exit)
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),

# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()
# CHECK-CPP: cb_reserve_back(
# CHECK-CPP: noc_async_read_tile(
# CHECK-CPP: noc_async_read_barrier();
# CHECK-CPP: cb_push_back(

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: cb_wait_front(
# CHECK-CPP: noc_async_write_tile(
# CHECK-CPP: noc_async_write_barrier();
# CHECK-CPP: cb_pop_front(


if __name__ == "__main__":
    import torch
    from test_helpers import require_hardware

    print("=== With-Pattern Add Kernel Test ===")

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

        print("Compiling with-pattern add kernel...")
        add_with_kernel(lhs, rhs, out)

        print("=== With-Pattern Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
