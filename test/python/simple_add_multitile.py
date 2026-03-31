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
Multi-tile add kernel - verifies correct tile indexing across 2x2 tile grid.

Uses 64x64 tensors (2x2 tiles of 32x32) to test that iter_index
correctly computes tile offsets in loops.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def add_multitile_kernel(lhs, rhs, out):
    """Add kernel processing 2x2 tile grid (4 tiles total)."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(2, 2), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        result = l + r
        o.store(result)
        l.pop()
        r.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        tx_lhs = ttl.copy(lhs[0:2, 0:2], lhs_blk)
        tx_lhs.wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        tx_rhs = ttl.copy(rhs[0:2, 0:2], rhs_blk)
        tx_rhs.wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Initial IR Checks - Verify compute kernel with multi-tile support
# =============================================================================

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# DFB operations (alphabetical order: lhs_cb=0, out_cb=2, rhs_cb=1)
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1

# Wait operations
# CHECK-DAG: ttl.cb_wait %[[CB0]]
# CHECK-DAG: ttl.cb_wait %[[CB1]]

# Reserve operation
# CHECK: ttl.cb_reserve %[[CB2]]

# Add operation
# CHECK: ttl.add

# Pop/push operations
# CHECK-DAG: ttl.cb_pop %[[CB0]]
# CHECK-DAG: ttl.cb_pop %[[CB1]]
# CHECK: ttl.cb_push %[[CB2]]

# CHECK-LABEL: func.func @dm_read

# =============================================================================
# C++ Kernel Checks - Verify loops are generated for multi-tile
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()

# Loop bound constant for 2x2 tile grid
# CHECK-CPP: size_t [[BOUND:v[0-9]+]] = 2;

# DFB operations before loops
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),
# CHECK-CPP: init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));

# Nested loops for 2x2 tile grid
# CHECK-CPP: for (size_t [[I:i[0-9]+]] = {{.*}}; [[I]] < [[BOUND]]; [[I]] += {{.*}}) {
# CHECK-CPP: for (size_t [[J:j[0-9]+]] = {{.*}}; [[J]] < [[BOUND]]; [[J]] += {{.*}}) {

# Linearized index calculation: i * 2 + j
# CHECK-CPP: size_t [[COLS:v[0-9]+]] = 2;
# CHECK-CPP: size_t [[ROW_OFF:v[0-9]+]] = [[I]] * [[COLS]];
# CHECK-CPP: size_t [[LIN_IDX:v[0-9]+]] = [[ROW_OFF]] + [[J]];

# Copy tiles using linearized index (at first use: CB0 then CB1)
# CHECK-CPP: copy_tile(get_compile_time_arg_val(0), [[LIN_IDX]],
# CHECK-CPP: copy_tile(get_compile_time_arg_val(1), [[LIN_IDX]],

# Add operation
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(

# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),

# =============================================================================
# FPU path checks (default: --ttl-maximize-dst --ttl-fpu-binary-ops)
# 2x2 = 4 tiles fits in DST (bf16), fully unrolled with FPU binary add
# =============================================================================

# CHECK-CPP-FPU: // add_compute
# CHECK-CPP-FPU: void kernel_main()
# CHECK-CPP-FPU: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP-FPU: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP-FPU: cb_reserve_back(get_compile_time_arg_val(2),
# CHECK-CPP-FPU: binary_op_init_common(get_compile_time_arg_val(0), get_compile_time_arg_val(1), get_compile_time_arg_val(2));
# CHECK-CPP-FPU: tile_regs_acquire();
# CHECK-CPP-FPU: add_tiles_init(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
# CHECK-CPP-FPU: add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),
# CHECK-CPP-FPU: add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),
# CHECK-CPP-FPU: add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),
# CHECK-CPP-FPU: add_tiles(get_compile_time_arg_val(0), get_compile_time_arg_val(1),
# CHECK-CPP-FPU: tile_regs_commit();
# CHECK-CPP-FPU: tile_regs_wait();
# CHECK-CPP-FPU: pack_tile_block(
# CHECK-CPP-FPU: tile_regs_release();
# CHECK-CPP-FPU: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP-FPU: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP-FPU: cb_push_back(get_compile_time_arg_val(2),


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Multi-tile Add Kernel Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        # 64x64 = 2x2 tiles of 32x32
        lhs_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((64, 64), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

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

        print("Compiling multi-tile add kernel (64x64 = 2x2 tiles)...")
        add_multitile_kernel(lhs, rhs, out)

        print("=== Multi-tile Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
