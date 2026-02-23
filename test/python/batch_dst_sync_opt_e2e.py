# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Multi-tile exp kernel (2x2 tiles, unary) - verifies batch DST sync optimization.

Uses 64x64 tensors (2x2 tiles of 32x32). Since exp is unary (dstPerIter=1)
and totalTrip=4 fits in DST capacity 8, the batch-dst-sync pass should unroll
the tile loop and emit a single acquire/release cycle.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def exp_multitile_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def exp_compute():
        i = inp_dfb.wait()
        o = out_dfb.reserve()
        result = ttl.math.exp(i)
        o.store(result)
        i.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[0:2, 0:2], inp_blk)
        tx_inp.wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0:2, 0:2])
        tx.wait()
        out_blk.pop()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @exp_compute
# CHECK: ttl.exp

# =============================================================================
# C++ Kernel Checks - batch DST sync: no loops, single acquire/release
# =============================================================================

# CHECK-CPP: // exp_compute
# CHECK-CPP: void kernel_main()

# CB operations before compute
# CHECK-CPP: cb_wait_front(get_compile_time_arg_val(0),
# CHECK-CPP: cb_reserve_back(get_compile_time_arg_val(1),

# Single acquire for all tiles (no loops)
# CHECK-CPP: tile_regs_acquire();

# Four copy+exp sequences (2x2 = 4 tiles), each with distinct dst index
# CHECK-CPP: copy_tile(
# CHECK-CPP: exp_tile_init();
# CHECK-CPP: exp_tile(
# CHECK-CPP: copy_tile(
# CHECK-CPP: exp_tile_init();
# CHECK-CPP: exp_tile(
# CHECK-CPP: copy_tile(
# CHECK-CPP: exp_tile_init();
# CHECK-CPP: exp_tile(
# CHECK-CPP: copy_tile(
# CHECK-CPP: exp_tile_init();
# CHECK-CPP: exp_tile(

# Single commit/wait for all tiles
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: tile_regs_wait();

# Four pack_tile calls
# CHECK-CPP: pack_tile<true>(
# CHECK-CPP: pack_tile<true>(
# CHECK-CPP: pack_tile<true>(
# CHECK-CPP: pack_tile<true>(

# Single release
# CHECK-CPP: tile_regs_release();


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Multi-tile Exp Kernel Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((64, 64), 1.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
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
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling multi-tile exp kernel (64x64 = 2x2 tiles)...")
        exp_multitile_kernel(inp, out)

        print("=== Multi-tile Exp Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
