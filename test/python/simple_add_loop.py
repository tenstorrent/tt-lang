# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s --no-ttl-maximize-dst --no-ttl-fpu-binary-ops > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.fpu.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP-FPU < %t.fpu.output

"""
Add kernel with explicit loop in compute - verifies for loops work inside kernels.

Uses a for loop to add the same values multiple times (accumulate pattern).
This tests loop support without requiring dynamic indices in data movement.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def add_loop_kernel(lhs, rhs, out):
    """Add kernel with loop in compute to accumulate results via `+=`."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def add_compute():
        with lhs_dfb.wait() as l, rhs_dfb.wait() as r:
            out_blk = out_dfb.reserve()
            out_blk.store(l)
            for i in range(4):
                out_blk += r
            out_blk.push()

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_blk:
            tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
            tx_lhs.wait()

        with rhs_dfb.reserve() as rhs_blk:
            tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0, 0])
            tx.wait()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}
# CHECK: %[[LHS:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[OUT:.+]] = ttl.bind_cb{cb_index = 2
# CHECK: %[[RHS:.+]] = ttl.bind_cb{cb_index = 1
# CHECK: ttl.cb_wait %[[LHS]]
# CHECK: ttl.cb_wait %[[RHS]]
# CHECK: ttl.cb_reserve %[[OUT]]
# CHECK: ttl.store
# CHECK: scf.for
# CHECK: ttl.store {{.*}} {accumulate}
# CHECK: ttl.cb_push %[[OUT]]
# CHECK: ttl.cb_pop %[[RHS]]
# CHECK: ttl.cb_pop %[[LHS]]

# =============================================================================
# C++ Kernel Checks - Verify for loop in generated compute code
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()
# CHECK-CPP-DAG: int32_t [[ZERO:v[0-9]+]] = 0;
# CHECK-CPP-DAG: int32_t [[ONE:v[0-9]+]] = 1;
# CHECK-CPP-DAG: experimental::CircularBuffer [[LHS:.*]](get_compile_time_arg_val(0));
# CHECK-CPP-DAG: experimental::CircularBuffer [[RHS:.*]](get_compile_time_arg_val(1));
# CHECK-CPP-DAG: experimental::CircularBuffer [[OUT:.*]](get_compile_time_arg_val(2));
# CHECK-CPP: [[LHS]].wait_front(
# CHECK-CPP: [[RHS]].wait_front(
# CHECK-CPP: [[OUT]].reserve_back(
# CHECK-CPP: init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();
# CHECK-CPP: pack_tile<true>({{.*}}, get_compile_time_arg_val(2),
# CHECK-CPP: tile_regs_release();
# CHECK-CPP: init_sfpu(get_compile_time_arg_val(1), get_compile_time_arg_val(2));
# CHECK-CPP: llk_pack_reconfig_l1_acc([[ONE]])
# CHECK-CPP: for (size_t {{.*}} < {{.*}};
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(1));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(1),
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();
# CHECK-CPP: pack_tile<true>({{.*}}, get_compile_time_arg_val(2),
# CHECK-CPP: tile_regs_release();
# CHECK-CPP: [[OUT]].push_back(
# CHECK-CPP: llk_pack_reconfig_l1_acc([[ZERO]])
# CHECK-CPP: [[RHS]].pop_front(
# CHECK-CPP: [[LHS]].pop_front(

# =============================================================================
# FPU path checks (default: --ttl-maximize-dst --ttl-fpu-binary-ops)
# =============================================================================

# CHECK-CPP-FPU: // add_compute
# CHECK-CPP-FPU: void kernel_main()
# CHECK-CPP-FPU-DAG: int32_t [[ZERO:v[0-9]+]] = 0;
# CHECK-CPP-FPU-DAG: int32_t [[ONE:v[0-9]+]] = 1;
# CHECK-CPP-FPU-DAG: experimental::CircularBuffer [[LHS:.*]](get_compile_time_arg_val(0));
# CHECK-CPP-FPU-DAG: experimental::CircularBuffer [[RHS:.*]](get_compile_time_arg_val(1));
# CHECK-CPP-FPU-DAG: experimental::CircularBuffer [[OUT:.*]](get_compile_time_arg_val(2));
# CHECK-CPP-FPU: [[LHS]].wait_front(
# CHECK-CPP-FPU: [[RHS]].wait_front(
# CHECK-CPP-FPU: [[OUT]].reserve_back(
# CHECK-CPP-FPU: init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
# CHECK-CPP-FPU: tile_regs_acquire();
# CHECK-CPP-FPU: copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP-FPU: copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP-FPU: tile_regs_commit();
# CHECK-CPP-FPU: tile_regs_wait();
# CHECK-CPP-FPU: pack_tile<true>({{.*}}, get_compile_time_arg_val(2),
# CHECK-CPP-FPU: tile_regs_release();
# CHECK-CPP-FPU: init_sfpu(get_compile_time_arg_val(1), get_compile_time_arg_val(2));
# CHECK-CPP-FPU: llk_pack_reconfig_l1_acc([[ONE]])
# CHECK-CPP-FPU: for (size_t {{.*}} < {{.*}};
# CHECK-CPP-FPU: tile_regs_acquire();
# CHECK-CPP-FPU: copy_tile_init(get_compile_time_arg_val(1));
# CHECK-CPP-FPU: copy_tile(get_compile_time_arg_val(1),
# CHECK-CPP-FPU: tile_regs_commit();
# CHECK-CPP-FPU: tile_regs_wait();
# CHECK-CPP-FPU: pack_tile<true>({{.*}}, get_compile_time_arg_val(2),
# CHECK-CPP-FPU: tile_regs_release();
# CHECK-CPP-FPU: [[OUT]].push_back(
# CHECK-CPP-FPU: llk_pack_reconfig_l1_acc([[ZERO]])
# CHECK-CPP-FPU: [[RHS]].pop_front(
# CHECK-CPP-FPU: [[LHS]].pop_front(


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
