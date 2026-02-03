# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple add kernel - verifies Python DSL lowers to correct TTL ops and C++ code.

Tests CB operations, add compute, and data movement patterns.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Reserve CB space before reading into it
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
        # Wait for data to be ready, then write out
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


# =============================================================================
# Initial IR Checks - TTNN layout attributes
# =============================================================================

# CHECK: #ttnn.buffer_type<l1>
# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>{{.*}}>

# =============================================================================
# Initial IR Checks - Verify TTL dialect ops (compute kernel)
# =============================================================================

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# Bind circular buffers (alphabetical order of capture names: lhs_cb, out_cb, rhs_cb)
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1

# Wait for input CBs
# CHECK: %[[L:.+]] = ttl.cb_wait %[[CB0]]
# CHECK: ttl.attach_cb %[[L]], %[[CB0]]
# CHECK: %[[R:.+]] = ttl.cb_wait %[[CB1]]
# CHECK: ttl.attach_cb %[[R]], %[[CB1]]

# Reserve output CB
# CHECK: ttl.cb_reserve %[[CB2]]
# CHECK: ttl.attach_cb %{{.+}}, %[[CB2]]

# Add operation (from l + r dunder method)
# CHECK: ttl.add

# Store result to output CB
# CHECK: ttl.tensor_store %{{.+}}, %[[CB2]]

# Finalize: pop inputs, push output
# CHECK: ttl.cb_pop %[[CB0]]
# CHECK: ttl.cb_pop %[[CB1]]
# CHECK: ttl.cb_push %[[CB2]]

# =============================================================================
# Initial IR Checks - Data movement kernels
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #ttnn_layout>
# CHECK-SAME: %arg1: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #ttnn_layout>
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0 : i32, 1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>}

# Bind CBs (alphabetical order: lhs_cb, rhs_cb)
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1

# First input: reserve, slice, copy, wait, push
# CHECK: ttl.cb_reserve %[[CB0]]
# CHECK: %[[SLICE0:.+]] = ttl.tensor_slice %arg0
# CHECK: %[[TX1:.+]] = ttl.copy %[[SLICE0]], %[[CB0]] : {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait %[[TX1]]
# CHECK: ttl.cb_push %[[CB0]]

# Second input: reserve, slice, copy, wait, push
# CHECK: ttl.cb_reserve %[[CB1]]
# CHECK: %[[SLICE1:.+]] = ttl.tensor_slice %arg1
# CHECK: %[[TX2:.+]] = ttl.copy %[[SLICE1]], %[[CB1]] : {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait %[[TX2]]
# CHECK: ttl.cb_push %[[CB1]]

# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #ttnn_layout>
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [2 : i32], ttl.kernel_thread = #ttkernel.thread<noc>}

# Wait for output CB, slice, copy to device, pop
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
# CHECK: ttl.cb_wait %[[CB2]]
# CHECK: %[[SLICE2:.+]] = ttl.tensor_slice %arg0
# CHECK: %[[TX:.+]] = ttl.copy %[[CB2]], %[[SLICE2]] : {{.*}} -> !ttl.transfer_handle<write>
# CHECK: ttl.wait %[[TX]]
# CHECK: ttl.cb_pop %[[CB2]]

# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()

# Wait for input CBs
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),

# Reserve output CB
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),

# DST register lifecycle
# CHECK-CPP: tile_regs_acquire();

# Load tiles into DST
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(1));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(1),

# Add operation
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(

# DST synchronization
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();

# Pack result
# CHECK-CPP: pack_tile<false>(

# Release regs
# CHECK-CPP: tile_regs_release();

# Pop inputs, push output
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),

# =============================================================================
# C++ Kernel Checks - Verify generated dm_read kernel
# =============================================================================

# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()

# First input: reserve CB, read tile, push CB
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(0),
# CHECK-CPP: auto {{.*}} = TensorAccessorArgs<3, 0>();
# CHECK-CPP: TensorAccessor{{.*}}= TensorAccessor(
# CHECK-CPP: get_write_ptr(get_compile_time_arg_val(0))
# CHECK-CPP: noc_async_read_tile(
# CHECK-CPP: noc_async_read_barrier();
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(0),

# Second input: reserve CB, read tile, push CB
# TensorAccessorArgs<CTA, CRTA>: CTA indexes static tensor metadata (is_sharded,
# is_dram, etc.) in compile-time args shared by all kernels. CTA layout is
# [3 CBs, then TAs], so tensor 1's metadata is at index 3+1=4. CRTA indexes
# buffer addresses in common runtime args, filtered per-kernel (1 = second arg).
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(1),
# CHECK-CPP: auto {{.*}} = TensorAccessorArgs<4, 1>();
# CHECK-CPP: TensorAccessor{{.*}}= TensorAccessor(
# CHECK-CPP: get_write_ptr(get_compile_time_arg_val(1))
# CHECK-CPP: noc_async_read_tile(
# CHECK-CPP: noc_async_read_barrier();
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(1),

# =============================================================================
# C++ Kernel Checks - Verify generated dm_write kernel
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()

# Wait for output CB, write tile, pop CB
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(2),
# CHECK-CPP: auto {{.*}} = TensorAccessorArgs<5, 0>();
# CHECK-CPP: TensorAccessor{{.*}}= TensorAccessor(
# CHECK-CPP: get_read_ptr(get_compile_time_arg_val(2))
# CHECK-CPP: noc_async_write_tile(
# CHECK-CPP: noc_async_write_barrier();
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(2),


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Add Kernel Test ===")
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

        print("Compiling add kernel...")
        add_kernel(lhs, rhs, out)

        print("=== Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
