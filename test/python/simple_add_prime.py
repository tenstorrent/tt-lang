# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""Elementwise add over an 11x1 (prime row dim) tile block.

Exercises remainder peeling in ttl-subblock-compute-for-dst: at the bf16 DST
budget of 8 the divisor heuristic cannot subdivide the prime row dim (11 > 8)
and leaves the whole block unsubblocked. The rescue then raises the row dim to
8, which does not divide 11, so the pass peels it into an 8x1 main block plus a
3x1 remainder. This test confirms the peeled IR lowers to a compute kernel.

Numeric correctness is covered by test_subblock_prime.py.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl

ROW_TILES = 11
COL_TILES = 1


@ttl.operation(grid=(1, 1))
def add_prime(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(
        lhs, shape=(ROW_TILES, COL_TILES), block_count=2
    )
    rhs_dfb = ttl.make_dataflow_buffer_like(
        rhs, shape=(ROW_TILES, COL_TILES), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(ROW_TILES, COL_TILES), block_count=2
    )

    @ttl.compute()
    def add_compute():
        lhs_blk = lhs_dfb.wait()
        rhs_blk = rhs_dfb.wait()
        out_blk = out_dfb.reserve()
        out_blk.store(lhs_blk + rhs_blk)
        lhs_blk.pop()
        rhs_blk.pop()
        out_blk.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        tx_lhs = ttl.copy(lhs[0:ROW_TILES, 0:COL_TILES], lhs_blk)
        tx_lhs.wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        tx_rhs = ttl.copy(rhs[0:ROW_TILES, 0:COL_TILES], rhs_blk)
        tx_rhs.wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[0:ROW_TILES, 0:COL_TILES])
        tx_out.wait()
        out_blk.pop()


# Initial IR: one 11x1 ttl.compute. Subblocking runs later in the device
# pipeline, so the peel is not visible in the initial dump.

# CHECK-LABEL: func.func @add_compute
# CHECK: ttl.add
# CHECK: ttl.store

# Compute kernel C++: the peel is visible as two DST regions, the rescued 8x1
# main subblock and the 3x1 remainder, together covering all 11 tiles. Peeled
# tile offsets are asserted structurally in subblock_prime.mlir.

# CHECK-CPP: === add_compute kernel written to
# CHECK-CPP: void kernel_main()
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP: add_tiles_init
# CHECK-CPP-COUNT-8: add_tiles(
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: pack_tile
# CHECK-CPP: tile_regs_release();
# CHECK-CPP: tile_regs_acquire();
# CHECK-CPP-COUNT-3: add_tiles(
# CHECK-CPP: tile_regs_commit();
# CHECK-CPP: pack_tile
# CHECK-CPP: tile_regs_release();


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)
    try:
        shape = (ROW_TILES * 32, COL_TILES * 32)

        def to_device(tensor):
            on_device = ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.to_memory_config(on_device, memory_config=ttnn.L1_MEMORY_CONFIG)

        lhs = to_device(torch.zeros(shape, dtype=torch.bfloat16))
        rhs = to_device(torch.zeros(shape, dtype=torch.bfloat16))
        out = to_device(torch.zeros(shape, dtype=torch.bfloat16))

        add_prime(lhs, rhs, out)
    finally:
        ttnn.close_device(device)
