# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# XFAIL: *
# https://github.com/tenstorrent/tt-lang/issues/164
# RUN: %python %s > %t.output 2>&1
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


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)])
def add_with_kernel(lhs, rhs, out):
    """Add kernel using 'with' pattern for automatic CB lifecycle."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @compute()
    def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # 'with' handles wait/reserve at entry, pop/push at exit
        with lhs_cb.wait() as l, rhs_cb.wait() as r, out_cb.reserve() as o:
            result = l + r
            o.store(result)
        # Automatic: out_cb.push(), rhs_cb.pop(), lhs_cb.pop() (reverse order)

    @datamovement()
    def dm_read(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        # 'with' for reserve/push pattern
        with lhs_cb.reserve() as lhs_block:
            tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
            tx_lhs.wait()
        # Automatic: lhs_cb.push()

        with rhs_cb.reserve() as rhs_block:
            tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
            tx_rhs.wait()
        # Automatic: rhs_cb.push()

    @datamovement()
    def dm_write(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # 'with' for wait/pop pattern
        with out_cb.wait() as out_block:
            tx = copy(out_cb, out_accessor[0, 0])
            tx.wait()
        # Automatic: out_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - Verify 'with' generates correct CB ops
# =============================================================================

# CHECK: #ttnn.buffer_type<l1>
# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>{{.*}}>

# CHECK-LABEL: func.func @add_compute
# CHECK-SAME: attributes {ttl.kernel_thread = #ttkernel.thread<compute>}

# CB binding
# CHECK: %[[CB0:.+]] = ttl.bind_cb{cb_index = 0
# CHECK: %[[CB1:.+]] = ttl.bind_cb{cb_index = 1
# CHECK: %[[CB2:.+]] = ttl.bind_cb{cb_index = 2

# 'with' entry: wait for inputs, reserve output
# CHECK: ttl.cb_wait %[[CB0]]
# CHECK: ttl.cb_wait %[[CB1]]
# CHECK: ttl.cb_reserve %[[CB2]]

# Add operation
# CHECK: ttl.add

# 'with' exit: push output, pop inputs (reverse order)
# CHECK: ttl.cb_push %[[CB2]]
# CHECK: ttl.cb_pop %[[CB1]]
# CHECK: ttl.cb_pop %[[CB0]]

# =============================================================================
# Initial IR Checks - Data movement with 'with' pattern
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: attributes {ttl.kernel_thread = #ttkernel.thread<noc>}

# First CB: reserve, copy, push
# CHECK: ttl.cb_reserve
# CHECK: ttl.copy {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait
# CHECK: ttl.cb_push

# Second CB: reserve, copy, push
# CHECK: ttl.cb_reserve
# CHECK: ttl.copy {{.*}} -> !ttl.transfer_handle<read>
# CHECK: ttl.wait
# CHECK: ttl.cb_push

# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: attributes {ttl.kernel_thread = #ttkernel.thread<noc>}

# Output CB: wait, copy, pop
# CHECK: ttl.cb_wait
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

    print("=== With-Pattern Add Kernel Test ===")

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
