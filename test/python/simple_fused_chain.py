# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: %python %s 2>&1 | FileCheck %s --check-prefix=CHECK-RUN

"""
Fused kernel with 20 chained ops - tests deep fusion chains with stable ops.

Uses bounded ops (sigmoid, tanh, relu, abs, neg) to avoid overflow.
Chain 1: sigmoid^5(a) - 5 ops, bounded [0,1]
Chain 2: tanh^5(b) - 5 ops, bounded [-1,1]
Chain 3: abs(neg(abs(neg(abs(c))))) - 5 ops
Final: (chain1 + chain2) * chain3 + relu(chain1 - chain2) - 5 ops
Total: 20 ops
"""

import os

from ttlang import ttl, make_circular_buffer_like, sigmoid, tanh, abs, neg, relu
from ttlang.ttl_api import Program
from ttlang.operators import copy

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def fused_chain_kernel(a, b, c, out):
    """Kernel with 20 chained ops - deep fusion test."""
    a_cb = make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def fused_compute():
        av = a_cb.wait()
        bv = b_cb.wait()
        cv = c_cb.wait()
        o = out_cb.reserve()
        # 20 ops using stable bounded functions
        # Chain 1: 5 sigmoid ops, output in [0,1]
        a_chain = sigmoid(sigmoid(sigmoid(sigmoid(sigmoid(av)))))
        # Chain 2: 5 tanh ops, output in [-1,1]
        b_chain = tanh(tanh(tanh(tanh(tanh(bv)))))
        # Chain 3: 5 ops (abs, neg alternating)
        c_chain = abs(neg(abs(neg(abs(cv)))))
        # Final: 5 ops (add, mul, sub, add, relu)
        temp1 = a_chain + b_chain
        temp2 = temp1 * c_chain
        temp3 = a_chain - b_chain
        result = temp2 + relu(temp3)
        o.store(result)
        a_cb.pop()
        b_cb.pop()
        c_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        a_cb.reserve()
        tx_a = copy(a[0, 0], a_cb)
        tx_a.wait()
        a_cb.push()

        b_cb.reserve()
        tx_b = copy(b[0, 0], b_cb)
        tx_b.wait()
        b_cb.push()

        c_cb.reserve()
        tx_c = copy(c[0, 0], c_cb)
        tx_c.wait()
        c_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(fused_compute, dm_read, dm_write)(a, b, c, out)


# =============================================================================
# Initial IR Checks - Verify fused TTL ops
# =============================================================================

# CHECK-LABEL: func.func @fused_compute
# CHECK-SAME: attributes {ttl.kernel_thread = #ttkernel.thread<compute>}

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
# CHECK-DAG: ttl.mul
# CHECK-DAG: ttl.sub

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
# CHECK-CPP-DAG: mul_binary_tile
# CHECK-CPP-DAG: sub_binary_tile

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
        # Chain 1: 5 sigmoid ops
        a_chain = torch.sigmoid(torch.sigmoid(torch.sigmoid(torch.sigmoid(torch.sigmoid(a_f)))))
        # Chain 2: 5 tanh ops
        b_chain = torch.tanh(torch.tanh(torch.tanh(torch.tanh(torch.tanh(b_f)))))
        # Chain 3: abs(neg(abs(neg(abs(c)))))
        c_chain = torch.abs(-torch.abs(-torch.abs(c_f)))
        # Final: (a_chain + b_chain) * c_chain + relu(a_chain - b_chain)
        temp1 = a_chain + b_chain
        temp2 = temp1 * c_chain
        temp3 = a_chain - b_chain
        expected = temp2 + torch.relu(temp3)

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
            # CHECK-RUN: PASS
        else:
            max_err = (result.float() - expected.float()).abs().max().item()
            print(f"FAIL: Max error = {max_err:.6f}")
            print(f"Result[0,0] = {result[0,0].item()}, Expected = {expected[0,0].item()}")

        print("=== Fused Chain Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
