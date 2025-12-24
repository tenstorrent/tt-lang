# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# XFAIL: *
# https://github.com/tenstorrent/tt-lang/issues/163
# RUN: %run-test %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Multi-tile add kernel - verifies correct tile indexing across 2x2 tile grid.

Uses 64x64 tensors (2x2 tiles of 32x32) to test that linearized_index
correctly computes tile offsets in loops.
"""

import os
os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang.ttl_api import (
    pykernel_gen,
    Program,
    CircularBuffer,
    TensorAccessor,
    compute,
    datamovement,
)
from ttlang.operators import copy

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(2, 2), (2, 2), (2, 2)])
def add_multitile_kernel(lhs, rhs, out):
    """Add kernel processing 2x2 tile grid (4 tiles total)."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @compute()
    def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_read(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_cb.reserve()
        tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @datamovement()
    def dm_write(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        out_cb.wait()
        tx = copy(out_cb, out_accessor[0, 0])
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

# CB operations
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2

# CHECK: ttl.cb_wait %[[CB0]]
# CHECK: ttl.cb_wait %[[CB1]]
# CHECK: ttl.cb_reserve %[[CB2]]

# Add operation
# CHECK: ttl.add

# CHECK: ttl.cb_pop %[[CB0]]
# CHECK: ttl.cb_pop %[[CB1]]
# CHECK: ttl.cb_push %[[CB2]]

# =============================================================================
# C++ Kernel Checks - Verify loops are generated for multi-tile
# =============================================================================

# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()

# Nested loops for 2x2 tile grid
# CHECK-CPP: for (size_t {{.*}} = 0; {{.*}} < 2;
# CHECK-CPP: for (size_t {{.*}} = 0; {{.*}} < 2;

# CB operations inside loop
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),

# Add operation
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(

# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),


if __name__ == "__main__":
    import torch

    print("=== Multi-tile Add Kernel Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        # 64x64 = 2x2 tiles of 32x32
        lhs_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((64, 64), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        lhs = ttnn.from_torch(lhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rhs = ttnn.from_torch(rhs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.from_torch(out_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        rhs = ttnn.to_memory_config(rhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling multi-tile add kernel (64x64 = 2x2 tiles)...")
        add_multitile_kernel(lhs, rhs, out)

        print("=== Multi-tile Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
