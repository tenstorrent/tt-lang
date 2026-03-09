# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Accumulation with a deep expression chain: out = exp((a + b) * c), acc=True.

Verifies that acc=True propagates correctly through a mixed unary/binary
fusion chain (add -> mul -> exp -> store {acc=true}).
"""

import os

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def acc_chain_kernel(a, b, c, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(4, 4), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(4, 4), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(4, 4), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(4, 4), buffer_factor=2)

    @ttl.compute()
    def acc_chain_compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv, c_dfb.wait() as cv:
            with out_dfb.reserve() as o:
                o.store(ttl.exp((av + bv) * cv), acc=True)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:4, 0:4], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:4, 0:4], blk)
            tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(c[0:4, 0:4], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:4, 0:4])
            tx.wait()


# =============================================================================
# Initial IR Checks - Verify TTL dialect ops with acc attribute
# =============================================================================

# CHECK-LABEL: func.func @acc_chain_compute
# CHECK-SAME: attributes {ttl.base_cta_index = {{[0-9]+}} : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# Wait for three inputs
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait

# Reserve output
# CHECK: ttl.cb_reserve

# Expression chain: add -> mul -> exp -> store {acc = true}
# CHECK: ttl.add
# CHECK: ttl.mul
# CHECK: ttl.exp
# CHECK: ttl.store {{.*}} {acc = true}

# Finalize
# CHECK: ttl.cb_push
# CHECK: ttl.cb_pop
# CHECK: ttl.cb_pop
# CHECK: ttl.cb_pop

# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel
# =============================================================================

# CHECK-CPP: // acc_chain_compute
# CHECK-CPP: void kernel_main()

# Wait for input CBs
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(2),

# Reserve output
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(3),

# DST register lifecycle
# CHECK-CPP: tile_regs_acquire();

# Load c into DST before FPU add (add overwrites its DST slot)
# CHECK-CPP: copy_tile_init(
# CHECK-CPP: copy_tile(

# FPU add from CBs: a + b
# CHECK-CPP: add_tiles_init(
# CHECK-CPP: add_tiles(

# SFPU mul: (a + b) * c — both operands in DST
# CHECK-CPP: mul_binary_tile_init();
# CHECK-CPP: mul_binary_tile(

# SFPU exp in-place
# CHECK-CPP: exp_tile_init();
# CHECK-CPP: exp_tile(

# DST synchronization and pack
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();
# CHECK-CPP: pack_tile<true>(
# CHECK-CPP: tile_regs_release();

# Push output, pop inputs (reverse order)
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(3),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(2),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Acc Chain Kernel Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        torch.manual_seed(42)
        a_torch = torch.rand((128, 128), dtype=torch.bfloat16) * 2 - 1  # [-1, 1]
        b_torch = torch.rand((128, 128), dtype=torch.bfloat16) * 2 - 1
        c_torch = torch.rand((128, 128), dtype=torch.bfloat16) * 2 - 1
        out_torch = torch.zeros((128, 128), dtype=torch.bfloat16)

        expected = torch.exp((a_torch + b_torch) * c_torch)

        to_dev = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        a = to_dev(a_torch)
        b = to_dev(b_torch)
        c = to_dev(c_torch)
        out = to_dev(out_torch)

        a = ttnn.to_memory_config(a, memory_config=ttnn.L1_MEMORY_CONFIG)
        b = ttnn.to_memory_config(b, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_memory_config(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling acc chain kernel (exp((a+b)*c), acc=True)...")
        acc_chain_kernel(a, b, c, out)

        if not os.environ.get("TTLANG_COMPILE_ONLY"):
            result = ttnn.to_torch(out)

            print(f"Input A sample: {a_torch[0, :5]}")
            print(f"Input B sample: {b_torch[0, :5]}")
            print(f"Input C sample: {c_torch[0, :5]}")
            print(f"Result sample:  {result[0, :5]}")
            print(f"Expected:       {expected[0, :5]}")

            assert torch.allclose(
                result, expected, rtol=0.05, atol=0.05
            ), f"Result mismatch! Max diff: {(result - expected).abs().max().item()}"
            print("[PASS] Results match expected values")

        print("=== Acc Chain Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
