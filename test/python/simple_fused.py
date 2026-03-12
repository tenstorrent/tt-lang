# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s --no-ttl-maximize-dst --no-ttl-fpu-binary-ops > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: env %python %s > %t.fpu.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP-FPU < %t.fpu.output

"""
Simple fused kernel - verifies ttl.math.exp(inp) + ttl.math.sqrt(bias) fusion lowers correctly.

Tests that multiple elementwise ops fuse into a single compute body.
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
def fused_kernel(inp, bias, out):
    """Kernel that computes ttl.math.exp(inp) + ttl.math.sqrt(bias) - fuses 3 ops."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fused_compute():
        i = inp_dfb.wait()
        b = bias_dfb.wait()
        o = out_dfb.reserve()
        result = ttl.math.exp(i) + ttl.math.sqrt(b)
        o.store(result)
        i.pop()
        b.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[0, 0], inp_blk)
        tx_inp.wait()
        inp_blk.push()

        bias_blk = bias_dfb.reserve()
        tx_bias = ttl.copy(bias[0, 0], bias_blk)
        tx_bias.wait()
        bias_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Initial IR Checks - Verify fused TTL ops in compute function
# =============================================================================

# CHECK-LABEL: func.func @fused_compute
# CHECK-SAME: attributes {ttl.base_cta_index = {{[0-9]+}} : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# Wait for inputs and reserve output
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve

# Fused operations: exp, sqrt, add
# CHECK: ttl.exp
# CHECK: ttl.sqrt
# CHECK: ttl.add

# Finalize
# CHECK: ttl.cb_pop
# CHECK: ttl.cb_pop
# CHECK: ttl.cb_push

# =============================================================================
# C++ Kernel Checks - Verify generated fused compute kernel
# =============================================================================

# CHECK-CPP: // fused_compute
# CHECK-CPP: void kernel_main()

# Wait for input CBs
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),

# Reserve output DFB
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),

# DST register lifecycle
# CHECK-CPP: tile_regs_acquire();

# Load tiles into DST at first use, interleaved with compute ops
# CB0 copy before exp (its first consumer)
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP: exp_tile_init();
# CHECK-CPP: exp_tile(

# CB1 copy before sqrt (its first consumer)
# CHECK-CPP: copy_tile_init(get_compile_time_arg_val(1));
# CHECK-CPP: copy_tile(get_compile_time_arg_val(1),
# CHECK-CPP: sqrt_tile_init();
# CHECK-CPP: sqrt_tile(

# Add operation
# CHECK-CPP: add_binary_tile_init();
# CHECK-CPP: add_binary_tile(

# DST synchronization
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();

# Pack result
# CHECK-CPP: pack_tile<true>(

# Release regs
# CHECK-CPP: tile_regs_release();

# Pop inputs, push output
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(2),

# =============================================================================
# FPU path checks (default: --ttl-maximize-dst --ttl-fpu-binary-ops)
# Add stays SFPU (both inputs from DST), but scheduling groups copy_tiles
# =============================================================================

# CHECK-CPP-FPU: // fused_compute
# CHECK-CPP-FPU: void kernel_main()
# CHECK-CPP-FPU: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP-FPU: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP-FPU: cb_reserve_back(get_compile_time_arg_val(2),
# CHECK-CPP-FPU: init_sfpu(get_compile_time_arg_val(0), get_compile_time_arg_val(2));
# CHECK-CPP-FPU: tile_regs_acquire();

# Scheduling groups copy_tiles together before SFPU ops
# CHECK-CPP-FPU: copy_tile_init(get_compile_time_arg_val(0));
# CHECK-CPP-FPU: copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP-FPU: copy_tile_init(get_compile_time_arg_val(1));
# CHECK-CPP-FPU: copy_tile(get_compile_time_arg_val(1),

# SFPU ops after copies
# CHECK-CPP-FPU: exp_tile_init();
# CHECK-CPP-FPU: exp_tile(
# CHECK-CPP-FPU: sqrt_tile_init();
# CHECK-CPP-FPU: sqrt_tile(
# CHECK-CPP-FPU: add_binary_tile_init();
# CHECK-CPP-FPU: add_binary_tile(
# CHECK-CPP-FPU: tile_regs_commit();
# CHECK-CPP-FPU: tile_regs_wait();
# CHECK-CPP-FPU: pack_tile<true>(
# CHECK-CPP-FPU: tile_regs_release();
# CHECK-CPP-FPU: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP-FPU: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP-FPU: cb_push_back(get_compile_time_arg_val(2),


if __name__ == "__main__":
    import torch

    print("=== Fused Kernel Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        bias_torch = torch.full((32, 32), 4.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        bias = ttnn.from_torch(
            bias_torch,
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

        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        bias = ttnn.to_memory_config(bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling fused kernel (exp + sqrt + add)...")
        fused_kernel(inp, bias, out)

        print("=== Fused Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
