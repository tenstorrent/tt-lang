# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s --ttl-auto-sync > %t.output 2>&1
# RUN: FileCheck %s < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Compile test: 8x8 matmul (64 output tiles) auto-subblocked for DST.

Exceeds bf16 DST capacity (8 tiles) by 8x. The compiler partitions this
into DST-sized subblocks automatically.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: Compiled kernel ready

# Verify subblock loop structure: 8 iterations, each computing one row
# of the 8x8 output via matmul_block and packing 8 tiles with
# pack_tile_block (enabled by per-subblock cb_reserve_back/cb_push_back).
# CHECK-CPP: void kernel_main()
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(1),
# CHECK-CPP: mm_block_init(
# CHECK-CPP: for (size_t {{.*}} = {{.*}}; {{.*}} < {{.*}}; {{.*}} += {{.*}}) {
# CHECK-CPP:   cb_reserve_back(get_compile_time_arg_val(2),
# CHECK-CPP:   tile_regs_acquire();
# CHECK-CPP:   experimental::matmul_block(
# CHECK-CPP:   tile_regs_commit();
# CHECK-CPP:   tile_regs_wait();
# CHECK-CPP:   pack_tile_block(
# CHECK-CPP:   tile_regs_release();
# CHECK-CPP:   cb_push_back(get_compile_time_arg_val(2),


@ttl.kernel(grid=(1, 1))
def matmul_8x8(a, b, y):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(8, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 8), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(8, 8), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
            with y_dfb.reserve() as y_blk:
                y_blk.store(a_blk @ b_blk)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:8, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0:8], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, y[0:8, 0:8])
            tx.wait()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_dram

    device = ttnn.open_device(device_id=0)
    try:
        make = lambda: to_dram(torch.randn(256, 256, dtype=torch.bfloat16), device)
        matmul_8x8(make(), make(), make())
    finally:
        ttnn.close_device(device)
