# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output.txt
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

"""
Test mixed L1/DRAM tensor configurations.

This test verifies that per-thread TensorAccessorArgs allows simple local indexing
when tensors have different memory configurations.

Configuration tested:
- lhs: L1 (interleaved)
- rhs: L1 (interleaved)
- out: DRAM (interleaved)

Validates that dm_write correctly uses TensorAccessorArgs<3, 0> (local index 0)
even though 'out' is the 3rd global tensor (global index 2).
"""

import torch
import ttnn
from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program


@ttl.kernel(grid=(1, 1))
def add_mixed_memory(lhs, rhs, out):
    """Add kernel with mixed L1 inputs and DRAM output."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
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
        # Read from L1 tensors
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

    return Program(compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - Verify mixed memory layout attributes
# =============================================================================

# L1 tensors
# CHECK: #[[L1:ttnn.buffer_type<l1>]]
# CHECK: #[[L1_LAYOUT:ttnn_layout.*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #[[L1]]>{{.*}}>

# DRAM tensors
# CHECK: #[[DRAM:ttnn.buffer_type<dram>]]
# CHECK: #[[DRAM_LAYOUT:ttnn_layout.*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, bf16>, #[[DRAM]]>{{.*}}>

# Compute thread (no tensor args)
# CHECK-LABEL: func.func @compute()
# CHECK-SAME: attributes {ttl.base_cta_index = [[COMPUTE_BASE:[0-9]+]] : i32, ttl.kernel_thread = #ttkernel.thread<compute>}

# dm_read thread (uses lhs and rhs from L1)
# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #[[L1_LAYOUT]]>
# CHECK-SAME: %arg1: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #[[L1_LAYOUT]]>
# CHECK-SAME: attributes {ttl.base_cta_index = [[DM_READ_BASE:[0-9]+]] : i32, ttl.kernel_thread = #ttkernel.thread<noc>}

# dm_write thread (uses out from DRAM)
# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: %arg0: tensor<{{[^>]+}}!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>
# CHECK-SAME: attributes {ttl.base_cta_index = [[DM_WRITE_BASE:[0-9]+]] : i32, ttl.kernel_thread = #ttkernel.thread<noc>}

# =============================================================================
# C++ Checks - dm_read with per-thread TensorAccessorArgs
# =============================================================================

# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()

# dm_read uses lhs and rhs (local indices 0, 1 in per-thread compile_time_args)
# CHECK-CPP: int32_t [[BANK_0:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
# CHECK-CPP: auto [[ACC1:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<3, 0>();
# CHECK-CPP: TensorAccessor [[TA1:v[0-9]+]] = TensorAccessor([[ACC1]], [[BANK_0]],
# CHECK-CPP: int32_t [[BANK_1:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
# CHECK-CPP: auto [[ACC2:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<4, 1>();
# CHECK-CPP: TensorAccessor [[TA2:v[0-9]+]] = TensorAccessor([[ACC2]], [[BANK_1]],

# =============================================================================
# C++ Checks - dm_write with per-thread TensorAccessorArgs
# =============================================================================

# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()

# dm_write uses only out (local index 0 in per-thread compile_time_args)
# Even though out is global tensor index 2, it uses TensorAccessorArgs<3, 0>
# because per-thread compile_time_args only contains out's config
# CHECK-CPP: int32_t [[BANK:v[0-9]+]] = get_common_arg_val<uint32_t>({{v[0-9]+}});
# CHECK-CPP: auto [[ACC:tensor_accessor_args_[0-9]+]] = TensorAccessorArgs<3, 0>();
# CHECK-CPP: TensorAccessor [[TA:v[0-9]+]] = TensorAccessor([[ACC]], [[BANK]],

# CHECK-OUTPUT: === Mixed L1/DRAM Test ===
print("=== Mixed L1/DRAM Test ===")

device = ttnn.open_device(device_id=0)

try:
    # Create test tensors
    lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
    rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
    out_torch = torch.full((32, 32), -1000.0, dtype=torch.bfloat16)
    expected = lhs_torch + rhs_torch  # Should be 5.0

    # Create DRAM tensors first
    lhs_dram = ttnn.from_torch(
        lhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rhs_dram = ttnn.from_torch(
        rhs_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_dram = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Move inputs to L1, keep output in DRAM
    # This creates the mixed memory configuration that exposes the bug
    lhs = ttnn.to_memory_config(lhs_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    rhs = ttnn.to_memory_config(rhs_dram, memory_config=ttnn.L1_MEMORY_CONFIG)
    out = out_dram  # Intentionally keep in DRAM

    # CHECK-OUTPUT: lhs: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1
    print(f"lhs: {lhs.memory_config()}")
    # CHECK-OUTPUT: rhs: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1
    print(f"rhs: {rhs.memory_config()}")
    # CHECK-OUTPUT: out: MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM
    print(f"out: {out.memory_config()}")

    # CHECK-OUTPUT: Running kernel
    print("Running kernel...")
    add_mixed_memory(lhs, rhs, out)

    # Verify results
    out_result = ttnn.to_torch(out)

    print(f"\nout[0:3, 0:3] =\n{out_result[0:3, 0:3]}")
    print(f"expected[0:3, 0:3] =\n{expected[0:3, 0:3]}")

    if torch.allclose(out_result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        # CHECK-OUTPUT: PASS
        print("\nPASS: Mixed L1/DRAM test passed!")
    else:
        max_err = (out_result.float() - expected.float()).abs().max().item()
        print(f"\nFAIL: Max error = {max_err:.6f}")
        print("This failure indicates TensorAccessorArgs is using wrong CTA index")

finally:
    ttnn.close_device(device)

# CHECK-OUTPUT: Test Complete
print("\n=== Test Complete ===")

# CHECK: Test Complete
print("\n=== Test Complete ===")
