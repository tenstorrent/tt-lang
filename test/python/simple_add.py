# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
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
from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program


@ttl.kernel(grid=(1, 1))
def add_kernel(lhs, rhs, out):
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

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
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        # Wait for data to be ready, then write out
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - TTNN layout attributes
# =============================================================================

# CHECK: #ttnn.buffer_type<l1>
# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>{{.*}}>

# =============================================================================
# Initial IR Checks - Verify TTL dialect ops (compute kernel)
# =============================================================================

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.base_cta_index = [[COMPUTE_BASE_CTA:[0-9]+]] : i32, ttl.kernel_thread = #ttkernel.thread<compute>}

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

# Attach result to output CB
# CHECK: ttl.attach_cb %{{.+}}, %[[CB2]]

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
# CHECK-SAME: attributes {ttl.base_cta_index = [[DM_READ_BASE_CTA:[0-9]+]] : i32, ttl.kernel_thread = #ttkernel.thread<noc>}

# Bind CBs (alphabetical order: lhs_cb, rhs_cb)
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1

# First input: reserve, copy, wait, push
# CHECK: ttl.cb_reserve %[[CB0]]
# CHECK: %[[TX1:.+]] = ttl.copy %arg0, %[[CB0]] : {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait %[[TX1]]
# CHECK: ttl.cb_push %[[CB0]]

# Second input: reserve, copy, wait, push
# CHECK: ttl.cb_reserve %[[CB1]]
# CHECK: %[[TX2:.+]] = ttl.copy %arg1, %[[CB1]] : {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait %[[TX2]]
# CHECK: ttl.cb_push %[[CB1]]

# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #ttnn_layout>
# CHECK-SAME: attributes {ttl.base_cta_index = [[DM_WRITE_BASE_CTA:[0-9]+]] : i32, ttl.kernel_thread = #ttkernel.thread<noc>}

# Only out_cb (index 2) is bound - only CBs declared in function signature are bound
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2

# Wait for output CB, copy to device, pop
# CHECK: ttl.cb_wait %[[CB2]]
# CHECK: %[[TX:.+]] = ttl.copy %[[CB2]], %arg0 : {{.*}} -> !ttl.transfer_handle<write>
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

# Variable declarations at function entry
# CHECK-CPP: int32_t [[TILE_IDX:v[0-9]+]] = 0;

# Accessors materialized at function entry
# Base CTA index is 3 (total number of CBs in the kernel: lhs_cb, rhs_cb, out_cb)
# First accessor uses base index (3) and runtime offset 0
# CHECK-CPP: int32_t [[BANK_0:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
# CHECK-CPP: auto [[ACC1_ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<3, 0>();
# CHECK-CPP: TensorAccessor [[TA1:v[0-9]+]] = TensorAccessor([[ACC1_ARGS]], [[BANK_0]],
# Second accessor uses incremented indices (base+1, runtime+1)
# CHECK-CPP: int32_t [[BANK_1:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
# CHECK-CPP: auto [[ACC2_ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<4, 1>();
# CHECK-CPP: TensorAccessor [[TA2:v[0-9]+]] = TensorAccessor([[ACC2_ARGS]], [[BANK_1]],

# First input: reserve CB, read tile using TA1, push CB
# CHECK-CPP: int32_t [[NTILES:v[0-9]+]] = 1;
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(0), [[NTILES]]);
# CHECK-CPP: int32_t [[PTR_0:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(0))
# CHECK-CPP: noc_async_read_tile([[TILE_IDX]], [[TA1]], [[PTR_0]]);
# CHECK-CPP: noc_async_read_barrier();
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(0), [[NTILES]]);

# Second input: reserve CB, read tile using TA2, push CB
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(1), [[NTILES]]);
# CHECK-CPP: int32_t [[PTR_1:v[0-9]+]] = get_write_ptr(get_compile_time_arg_val(1))
# CHECK-CPP: noc_async_read_tile([[TILE_IDX]], [[TA2]], [[PTR_1]]);
# CHECK-CPP: noc_async_read_barrier();
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(1), [[NTILES]]);


# =============================================================================
# C++ Kernel Checks - Verify generated dm_write kernel
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()

# Variable declarations at function entry
# CHECK-CPP: int32_t [[NTILES_W:v[0-9]+]] = 1;
# CHECK-CPP: int32_t [[TILE_IDX_W:v[0-9]+]] = 0;

# Accessor materialized at function entry
# Base CTA index is 3 (total number of CBs in the kernel)
# dm_write uses 1 tensor (out), gets buffer address from common_runtime_args[0]
# CHECK-CPP: int32_t [[BANK:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
# CHECK-CPP: auto [[ACC_ARGS:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<3, 0>();
# CHECK-CPP: TensorAccessor [[TA:v[0-9]+]] = TensorAccessor([[ACC_ARGS]], [[BANK]],

# Wait for output CB (cb_index = 2), write tile using TA, pop CB
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(2), [[NTILES_W]]);
# CHECK-CPP: int32_t [[PTR:v[0-9]+]] = get_read_ptr(get_compile_time_arg_val(2))
# CHECK-CPP: noc_async_write_tile([[TILE_IDX_W]], [[TA]], [[PTR]]);
# CHECK-CPP: noc_async_write_barrier();
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(2), [[NTILES_W]]);


if __name__ == "__main__":
    import torch
    from utils import require_hardware

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
