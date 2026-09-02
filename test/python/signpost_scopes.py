# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s --no-ttl-maximize-dst --no-ttl-fpu-binary-ops > %t.output 2>&1
# RUN: FileCheck %s < %t.output
# RUN: %python %s > %t.fpu.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-FPU < %t.fpu.output

"""
Broadcast multitile blocks kernel - verifies user-defined signpost scopes
and nested loop structure with broadcast/math ops in generated C++ compute
kernel.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

TILE_SIZE = 32
GRANULARITY = 4


@ttl.operation(grid=(1, 1))
def bcast_multitile_kernel(
    a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor
):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    rows = y.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = y.shape[1] // TILE_SIZE // col_tiles_per_block

    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(row_tiles_per_block, 1), block_count=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(1, col_tiles_per_block), block_count=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(row_tiles_per_block, col_tiles_per_block), block_count=2
    )

    @ttl.compute()
    def demo_compute():
        with c_dfb.wait() as c_blk:
            for _ in range(rows):
                for _ in range(cols):
                    with (
                        a_dfb.wait() as a_blk,
                        b_dfb.wait() as b_blk,
                        y_dfb.reserve() as y_blk,
                    ):
                        with ttl.signpost("compute"):
                            with ttl.signpost("broadcast"):
                                a_bcast = ttl.block.broadcast(
                                    a_blk,
                                    dims=[1],
                                    shape=(row_tiles_per_block, col_tiles_per_block),
                                )
                                b_bcast = ttl.block.broadcast(
                                    b_blk,
                                    dims=[0],
                                    shape=(row_tiles_per_block, col_tiles_per_block),
                                )
                                c_bcast = ttl.block.broadcast(
                                    c_blk,
                                    dims=[0, 1],
                                    shape=(row_tiles_per_block, col_tiles_per_block),
                                )
                                with ttl.signpost("math"):
                                    tmp = a_bcast * b_bcast + c_bcast
                                    with ttl.signpost("store"):
                                        y_blk.store(tmp)

    @ttl.datamovement()
    def demo_read():
        with c_dfb.reserve() as c_block:
            ttl.copy(c[0, 0], c_block).wait()
        for row in range(rows):
            row_begin = row * row_tiles_per_block
            row_end = row_begin + row_tiles_per_block
            for col in range(cols):
                col_begin = col * col_tiles_per_block
                col_end = col_begin + col_tiles_per_block
                with a_dfb.reserve() as a_block:
                    ttl.copy(a[row_begin:row_end, 0:1], a_block).wait()
                with b_dfb.reserve() as b_block:
                    ttl.copy(b[0:1, col_begin:col_end], b_block).wait()

    @ttl.datamovement()
    def demo_write():
        for row in range(rows):
            row_begin = row * row_tiles_per_block
            row_end = row_begin + row_tiles_per_block
            for col in range(cols):
                col_begin = col * col_tiles_per_block
                col_end = col_begin + col_tiles_per_block
                with y_dfb.wait() as y_block:
                    ttl.copy(y_block, y[row_begin:row_end, col_begin:col_end]).wait()


# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel structure
# =============================================================================

# CHECK: === demo_compute kernel written to {{.*}} ===
# CHECK: void kernel_main()

# No signpost scopes outside the inner tile loops
# CHECK-NOT:  DeviceZoneScopedN(
# CHECK:      init_sfpu(
# CHECK:      for (size_t [[K:.*]] = [[V6:.*]]; [[K]] < [[V4:.*]]; [[K]] += [[V5:.*]]) {
# CHECK-NEXT:   for (size_t [[L:.*]] = [[V6]]; [[L]] < [[V4]]; [[L]] += [[V5]]) {
# CHECK-NEXT:     tile_regs_acquire();
# CHECK-NEXT:     {
# CHECK-NEXT:     DeviceZoneScopedN("ttl_compute");
# CHECK-NEXT:     {
# CHECK-NEXT:     DeviceZoneScopedN("ttl_broadcast");
# CHECK-NEXT:     unary_bcast_init<BroadcastType::COL>(get_compile_time_arg_val(0), get_compile_time_arg_val(3));
# CHECK-NEXT:     unary_bcast<BroadcastType::COL>(get_compile_time_arg_val(0), [[K]], [[V6]]);
# CHECK-NEXT:     unary_bcast_init<BroadcastType::ROW>(get_compile_time_arg_val(1), get_compile_time_arg_val(3));
# CHECK-NEXT:     unary_bcast<BroadcastType::ROW>(get_compile_time_arg_val(1), [[L]], [[V5]]);
# CHECK-NEXT:     mul_binary_tile_init();
# CHECK-NEXT:     mul_binary_tile([[V6]], [[V5]], [[V6]]);
# CHECK-NEXT:     unary_bcast_init<BroadcastType::SCALAR>(get_compile_time_arg_val(2), get_compile_time_arg_val(3));
# CHECK-NEXT:     unary_bcast<BroadcastType::SCALAR>(get_compile_time_arg_val(2), [[V6]], [[V5]]);
# CHECK-NEXT:     {
# CHECK-NEXT:     DeviceZoneScopedN("ttl_math");
# CHECK-NEXT:     add_binary_tile_init();
# CHECK-NEXT:     add_binary_tile([[V6]], [[V5]], [[V6]]);
# CHECK-NEXT:     {
# CHECK-NEXT:     DeviceZoneScopedN("ttl_store");
# CHECK-NEXT:     tile_regs_commit();
# CHECK-NEXT:     tile_regs_wait();
# CHECK-NEXT:     size_t [[V12:.*]] = 4;
# CHECK-NEXT:     size_t [[V13:.*]] = [[K]] * [[V12]];
# CHECK-NEXT:     size_t [[V14:.*]] = [[V13]] + [[L]];
# CHECK-NEXT:     pack_tile<true>([[V6]], get_compile_time_arg_val(3), [[V14]]);
# CHECK-NEXT:     }
# CHECK-NEXT:     }
# CHECK-NEXT:     }
# CHECK-NEXT:     }
# CHECK-NEXT:     tile_regs_release();
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NOT:  DeviceZoneScopedN(
# CHECK: === demo_read kernel written to {{.*}} ===

# =============================================================================
# FPU path checks (default: --ttl-maximize-dst --ttl-fpu-binary-ops)
# User scopes preserve each of the four unrolled tile instances. Scheduling
# remains enabled within each scope but cannot move operations between scopes.
# =============================================================================

# CHECK-FPU: === demo_compute kernel written to {{.*}} ===
# CHECK-FPU: void kernel_main()

# No signpost scopes outside the inner subblock loop
# CHECK-FPU-NOT:  DeviceZoneScopedN(
# CHECK-FPU:      init_sfpu(
# CHECK-FPU:      for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-FPU-NEXT:   tile_regs_acquire();
# CHECK-FPU-NEXT:   {
# CHECK-FPU-NEXT:   DeviceZoneScopedN("ttl_compute");
# CHECK-FPU-NEXT:   {
# CHECK-FPU-NEXT:   DeviceZoneScopedN("ttl_broadcast");
# CHECK-FPU-NEXT:   unary_bcast_init<BroadcastType::COL>(
# CHECK-FPU-NEXT:   unary_bcast<BroadcastType::COL>(
# CHECK-FPU-NEXT:   unary_bcast_init<BroadcastType::ROW>(
# CHECK-FPU-NEXT:   unary_bcast<BroadcastType::ROW>(
# CHECK-FPU-NEXT:   mul_binary_tile_init();
# CHECK-FPU-NEXT:   mul_binary_tile(
# CHECK-FPU-NEXT:   unary_bcast_init<BroadcastType::SCALAR>(
# CHECK-FPU-NEXT:   unary_bcast<BroadcastType::SCALAR>(
# CHECK-FPU-NEXT:   {
# CHECK-FPU-NEXT:   DeviceZoneScopedN("ttl_math");
# CHECK-FPU-NEXT:   add_binary_tile_init();
# CHECK-FPU-NEXT:   add_binary_tile(
# CHECK-FPU-NEXT:   }
# CHECK-FPU-NEXT:   }
# CHECK-FPU-NEXT:   }

# The remaining three scoped tile instances precede the shared sync and pack.
# CHECK-FPU:        DeviceZoneScopedN("ttl_compute");
# CHECK-FPU:        DeviceZoneScopedN("ttl_broadcast");
# CHECK-FPU:        DeviceZoneScopedN("ttl_math");
# CHECK-FPU:        DeviceZoneScopedN("ttl_compute");
# CHECK-FPU:        DeviceZoneScopedN("ttl_broadcast");
# CHECK-FPU:        DeviceZoneScopedN("ttl_math");
# CHECK-FPU:        DeviceZoneScopedN("ttl_compute");
# CHECK-FPU:        DeviceZoneScopedN("ttl_broadcast");
# CHECK-FPU:        DeviceZoneScopedN("ttl_math");
# CHECK-FPU:        tile_regs_commit();
# CHECK-FPU-NEXT:   tile_regs_wait();
# CHECK-FPU:        pack_tile<true>(
# CHECK-FPU:        pack_tile<true>(
# CHECK-FPU:        pack_tile<true>(
# CHECK-FPU:        pack_tile<true>(
# CHECK-FPU-NEXT:   tile_regs_release();
# CHECK-FPU-NEXT: }
# CHECK-FPU-NOT:  DeviceZoneScopedN(
# CHECK-FPU: === demo_read kernel written to {{.*}} ===


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        shape = (2048, 2048)

        a = torch.rand((shape[0], 1), dtype=torch.bfloat16)
        b = torch.rand((1, shape[1]), dtype=torch.bfloat16)
        c = torch.rand((1, 1), dtype=torch.bfloat16)

        y_torch = torch.zeros(shape, dtype=torch.bfloat16)

        def from_torch(tensor):
            return ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        a_tt = from_torch(a)
        b_tt = from_torch(b)
        c_tt = from_torch(c)
        y_tt = from_torch(y_torch)

        bcast_multitile_kernel(a_tt, b_tt, c_tt, y_tt)

    finally:
        ttnn.close_device(device)
