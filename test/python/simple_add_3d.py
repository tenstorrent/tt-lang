# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s --no-ttl-maximize-dst --no-ttl-fpu-binary-ops > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.fpu.block.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP-FPU-BLOCK < %t.fpu.block.output
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s --no-ttl-combine-pack-tiles > %t.fpu.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP-FPU < %t.fpu.output

"""
3D add kernel with multi-tile CB - verifies ND shape support in TTL ops.

Uses a 3D tensor (batch=2, 64x64) with CB shape (2, 2, 2) to test
multi-dimensional tensor slicing and CB indexing.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def add_3d_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(2, 2, 2), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(2, 2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2, 2), buffer_factor=2)

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
        tx_lhs = ttl.copy(lhs[0:2, 0:2, 0:2], lhs_blk)
        tx_lhs.wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        tx_rhs = ttl.copy(rhs[0:2, 0:2, 0:2], rhs_blk)
        tx_rhs.wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2, 0:2])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Initial IR Checks - 3D layout
# =============================================================================

# =============================================================================
# Compute kernel: 3D CB types and tensor ops
# =============================================================================

# CHECK-LABEL: func.func @add_compute

# 3D CB shapes
# CHECK: ttl.bind_cb{cb_index = 0, buffer_factor = 2} : <[2, 2, 2], !ttcore.tile<32x32, bf16>, 2>
# CHECK: ttl.bind_cb{cb_index = 2, buffer_factor = 2} : <[2, 2, 2], !ttcore.tile<32x32, bf16>, 2>
# CHECK: ttl.bind_cb{cb_index = 1, buffer_factor = 2} : <[2, 2, 2], !ttcore.tile<32x32, bf16>, 2>

# Wait/reserve produce 3D tensors of tiles
# CHECK: ttl.cb_wait %{{.*}} : <[2, 2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x2x!ttcore.tile<32x32, bf16>>
# CHECK: ttl.cb_wait %{{.*}} : <[2, 2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x2x!ttcore.tile<32x32, bf16>>
# CHECK: ttl.cb_reserve %{{.*}} : <[2, 2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x2x!ttcore.tile<32x32, bf16>>

# 3D eltwise add and store
# CHECK: ttl.add
# CHECK: ttl.store

# =============================================================================
# Data movement: 3D tensor_slice with 3 indices
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: %arg0: tensor<2x2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [2, 64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
# CHECK-SAME: %arg1: tensor<2x2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [2, 64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>

# tensor_slice with 3 indices on first tensor
# CHECK: ttl.tensor_slice %arg0[%{{.*}}, %{{.*}}, %{{.*}}] : tensor<2x2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [2, 64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>
# CHECK: ttl.copy

# tensor_slice with 3 indices on second tensor
# CHECK: ttl.tensor_slice %arg1[%{{.*}}, %{{.*}}, %{{.*}}] : tensor<2x2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [2, 64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>

# CHECK-LABEL: func.func @dm_write
# CHECK-SAME: %arg0: tensor<2x2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [2, 64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>

# tensor_slice with 3 indices on output
# CHECK: ttl.tensor_slice %arg0[%{{.*}}, %{{.*}}, %{{.*}}] : tensor<2x2x2x!ttcore.tile<32x32, bf16>, #ttl.layout<shape = [2, 64, 64], element_type = !ttcore.tile<32x32, bf16>, buffer = l1, grid = [1, 1], memory = interleaved>>

# =============================================================================
# C++ Kernel Checks - Verify 3D loop nests in generated code
# =============================================================================

# Compute kernel: 3 nested loops over 2x2x2 tile grid
# CHECK-CPP: // add_compute
# CHECK-CPP: void kernel_main()
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(2),
# CHECK-CPP: for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:   for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:     for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:       tile_regs_acquire();
# CHECK-CPP:       copy_tile(get_compile_time_arg_val(0),
# CHECK-CPP:       copy_tile(get_compile_time_arg_val(1),
# CHECK-CPP:       add_binary_tile(
# CHECK-CPP:       tile_regs_commit();
# CHECK-CPP:       tile_regs_wait();
# CHECK-CPP:       pack_tile<true>(
# CHECK-CPP:       tile_regs_release();

# DM read kernel: 3 nested loops with noc_async_read_tile
# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(0),
# CHECK-CPP: for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:   for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:     for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:       noc_async_read_tile(

# DM write kernel: 3 nested loops with noc_async_write_tile
# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(2),
# CHECK-CPP: for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:   for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:     for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:       noc_async_write_tile(

# =============================================================================
# FPU path checks (default: --ttl-maximize-dst --ttl-fpu-binary-ops)
# 2x2x2 = 8 tiles fits in DST (bf16), fully unrolled with FPU binary add
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
# CHECK-CPP-FPU: tile_regs_commit();
# CHECK-CPP-FPU: tile_regs_wait();
# CHECK-CPP-FPU: pack_tile<true>(
# CHECK-CPP-FPU: tile_regs_release();
# CHECK-CPP-FPU: cb_push_back(get_compile_time_arg_val(2),
# CHECK-CPP-FPU-NOT: pack_tile_block(

# Default (combine-pack-tiles enabled): individual pack_tile ops combined.
# CHECK-CPP-FPU-BLOCK: tile_regs_wait();
# CHECK-CPP-FPU-BLOCK: pack_tile_block(
# CHECK-CPP-FPU-BLOCK: tile_regs_release();
# CHECK-CPP-FPU-BLOCK: cb_push_back(get_compile_time_arg_val(2),
# CHECK-CPP-FPU-BLOCK-NOT: pack_tile<true>(


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== 3D Add Kernel Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        # 3D tensors: batch=2, 64x64 spatial (2x2 tiles per batch)
        lhs_torch = torch.full((2, 64, 64), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((2, 64, 64), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((2, 64, 64), dtype=torch.bfloat16)

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

        print("Compiling 3D add kernel...")
        add_3d_kernel(lhs, rhs, out)

        print("=== 3D Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
