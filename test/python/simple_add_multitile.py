# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Multi-tile add kernel - verifies correct tile indexing across 2x2 tile grid.

Uses 64x64 tensors (2x2 tiles of 32x32) to test that linearized_index
correctly computes tile offsets in loops.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program


@ttl.kernel(grid=(1, 1))
def add_multitile_kernel(lhs, rhs, out):
    """Add kernel processing 2x2 tile grid (4 tiles total)."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(2, 2), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(2, 2), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

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
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - Verify 2x2 block factors in tensor shapes
# =============================================================================

# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<2x2x!ttcore.tile<32x32, bf16>{{.*}}>

# =============================================================================
# Initial IR Checks - Verify compute kernel with multi-tile support
# =============================================================================

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.kernel_thread = #ttkernel.thread<compute>}

# CB operations (alphabetical order: lhs_cb=0, out_cb=2, rhs_cb=1)
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

# CB operations before loops
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),

# Nested loops for 2x2 tile grid
# CHECK-CPP: for (size_t [[I:i[0-9]+]] = {{.*}}; [[I]] < [[BOUND]]; [[I]] += {{.*}}) {
# CHECK-CPP: for (size_t [[J:j[0-9]+]] = {{.*}}; [[J]] < [[BOUND]]; [[J]] += {{.*}}) {

# Linearized index calculation: i * 2 + j
# CHECK-CPP: size_t [[COLS:v[0-9]+]] = 2;
# CHECK-CPP: size_t [[ROW_OFF:v[0-9]+]] = [[I]] * [[COLS]];
# CHECK-CPP: size_t [[LIN_IDX:v[0-9]+]] = [[ROW_OFF]] + [[J]];

# Copy tiles using linearized index
# CHECK-CPP: copy_tile(get_compile_time_arg_val(0), [[LIN_IDX]],
# CHECK-CPP: copy_tile(get_compile_time_arg_val(1), [[LIN_IDX]],

# Add operation
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(

# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),


if __name__ == "__main__":
    import torch
    from utils import require_hardware

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
