# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

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


@ttl.kernel(grid=(1, 1))
def bcast_multitile_kernel(
    a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor
):
    row_tiles_per_block = GRANULARITY
    col_tiles_per_block = GRANULARITY

    rows = y.shape[0] // TILE_SIZE // row_tiles_per_block
    cols = y.shape[1] // TILE_SIZE // col_tiles_per_block

    a_dfb = ttl.make_dataflow_buffer_like(
        a, shape=(row_tiles_per_block, 1), buffer_factor=2
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b, shape=(1, col_tiles_per_block), buffer_factor=2
    )
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(
        y, shape=(row_tiles_per_block, col_tiles_per_block), buffer_factor=2
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
                                a_bcast = ttl.math.broadcast(a_blk, y_blk, dims=[1])
                                b_bcast = ttl.math.broadcast(b_blk, y_blk, dims=[0])
                                c_bcast = ttl.math.broadcast(c_blk, y_blk, dims=[0, 1])
                                with ttl.signpost("math"):
                                    tmp = a_bcast * b_bcast + c_bcast
                                    with ttl.signpost("store"):
                                        y_blk.store(tmp)

    @ttl.datamovement()
    def demo_read():
        pass

    @ttl.datamovement()
    def demo_write():
        pass


# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel structure
# =============================================================================

# CHECK: // demo_compute
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
# CHECK-NEXT:     size_t [[V12:.*]] = [[K]] * [[V4]];
# CHECK-NEXT:     size_t [[V13:.*]] = [[V12]] + [[L]];
# CHECK-NEXT:     pack_tile<true>([[V6]], get_compile_time_arg_val(3), [[V13]]);
# CHECK-NEXT:     }
# CHECK-NEXT:     }
# CHECK-NEXT:     }
# CHECK-NEXT:     }
# CHECK-NEXT:     tile_regs_release();
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NOT:  DeviceZoneScopedN(


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
