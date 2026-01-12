# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: %python %s 2>&1 | FileCheck %s --check-prefix=EXEC

"""
Fused kernel with 20 chained ops - tests deep fusion chains with stable ops.

Uses bounded ops (sigmoid, tanh, relu, abs, neg) to avoid overflow.
Sequential chain avoids multiple uses of intermediate values.
"""

import os

from ttlang import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def fused_chain_kernel(a, b, c, out):
    """Kernel with 20 chained ops - deep fusion test."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fused_compute():
        av = a_cb.wait()
        bv = b_cb.wait()
        cv = c_cb.wait()
        o = out_cb.reserve()
        # 20 ops: sequential chain to avoid multiple uses of intermediates
        # Start with a: 5 unary ops
        v = ttl.math.sigmoid(av)  # 1
        v = ttl.math.sigmoid(v)  # 2
        v = ttl.math.tanh(v)  # 3
        v = ttl.math.tanh(v)  # 4
        v = ttl.math.abs(v)  # 5
        # Mix in b: 5 ops (1 binary + 4 unary)
        v = v + bv  # 6
        v = ttl.math.sigmoid(v)  # 7
        v = ttl.math.tanh(v)  # 8
        v = ttl.math.neg(v)  # 9
        v = ttl.math.abs(v)  # 10
        # Mix in c: 5 ops (1 binary + 4 unary)
        v = v + cv  # 11
        v = ttl.math.relu(v)  # 12
        v = ttl.math.sigmoid(v)  # 13
        v = ttl.math.tanh(v)  # 14
        v = ttl.math.abs(v)  # 15
        # Final: 5 more unary ops
        v = ttl.math.sigmoid(v)  # 16
        v = ttl.math.tanh(v)  # 17
        v = ttl.math.relu(v)  # 18
        v = ttl.math.sigmoid(v)  # 19
        result = ttl.math.tanh(v)  # 20
        o.store(result)
        a_cb.pop()
        b_cb.pop()
        c_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_cb.push()

        c_blk = c_cb.reserve()
        tx_c = ttl.copy(c[0, 0], c_blk)
        tx_c.wait()
        c_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(fused_compute, dm_read, dm_write)(a, b, c, out)


# =============================================================================
# Initial IR Checks - Verify fused TTL ops
# =============================================================================

# CHECK-LABEL: func.func @fused_compute
# CHECK-SAME: attributes {ttl.base_cta_index = {{[0-9]+}} : i32, ttl.crta_indices = [], ttl.kernel_thread = #ttkernel.thread<compute>}

# Wait for inputs and reserve output
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve

# Verify key ops appear (20-op chain with diverse ops)
# CHECK-DAG: ttl.sigmoid
# CHECK-DAG: ttl.tanh
# CHECK-DAG: ttl.abs
# CHECK-DAG: ttl.neg
# CHECK-DAG: ttl.relu
# CHECK-DAG: ttl.add

# Finalize
# CHECK: ttl.cb_pop
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
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(2),

# Reserve output CB
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(3),

# DST register lifecycle
# CHECK-CPP: tile_regs_acquire();

# Verify key tile ops appear (20-op chain with diverse ops)
# CHECK-CPP-DAG: sigmoid_tile
# CHECK-CPP-DAG: tanh_tile
# CHECK-CPP-DAG: abs_tile
# CHECK-CPP-DAG: negative_tile
# CHECK-CPP-DAG: relu_tile
# CHECK-CPP-DAG: add_binary_tile

# DST synchronization
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();

# Pack result
# CHECK-CPP: pack_tile<false>(

# Pop inputs, push output
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_pop_front(get_compile_time_arg_val(2),
# CHECK-CPP: cb_push_back(get_compile_time_arg_val(3),


if __name__ == "__main__":
    import torch

    print("=== Fused Chain Kernel Test (20 ops) ===")

    device = ttnn.open_device(device_id=0)

    try:
        # Use values that work well with bounded ops
        a_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        b_torch = torch.full((32, 32), 0.5, dtype=torch.bfloat16)
        c_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Compute expected result with torch (same ops as kernel)
        a_f = a_torch.float()
        b_f = b_torch.float()
        c_f = c_torch.float()
        # 20 sequential ops matching the kernel
        v = torch.sigmoid(a_f)  # 1
        v = torch.sigmoid(v)  # 2
        v = torch.tanh(v)  # 3
        v = torch.tanh(v)  # 4
        v = torch.abs(v)  # 5
        v = v + b_f  # 6
        v = torch.sigmoid(v)  # 7
        v = torch.tanh(v)  # 8
        v = -v  # 9 (neg)
        v = torch.abs(v)  # 10
        v = v + c_f  # 11
        v = torch.relu(v)  # 12
        v = torch.sigmoid(v)  # 13
        v = torch.tanh(v)  # 14
        v = torch.abs(v)  # 15
        v = torch.sigmoid(v)  # 16
        v = torch.tanh(v)  # 17
        v = torch.relu(v)  # 18
        v = torch.sigmoid(v)  # 19
        expected = torch.tanh(v)  # 20

        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        c = ttnn.from_torch(
            c_torch,
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

        a = ttnn.to_memory_config(a, memory_config=ttnn.L1_MEMORY_CONFIG)
        b = ttnn.to_memory_config(b, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_memory_config(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Running fused chain kernel (20 ops)...")
        fused_chain_kernel(a, b, c, out)

        # Verify result
        result = ttnn.to_torch(out)
        print(result)
        if torch.allclose(result.float(), expected.float(), rtol=1e-1, atol=1e-1):
            print("PASS: Output matches expected!")
            # EXEC: PASS
        else:
            max_err = (result.float() - expected.float()).abs().max().item()
            print(f"FAIL: Max error = {max_err:.6f}")
            print(
                f"Result[0,0] = {result[0,0].item()}, Expected = {expected[0,0].item()}"
            )

        print("=== Fused Chain Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
