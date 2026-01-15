# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Multicore kernel lit test - verifies core(dims=2) lowers to
get_absolute_logical_x() and get_absolute_logical_y() in generated C++.

Tests an 8x8 grid kernel that uses dynamic core indices for tile indexing.
Each core processes one tile from a 256x256 tensor (8x8 tiles).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(8, 8))
def multicore_add(lhs, rhs, out):
    """Multicore add kernel - each core processes its own tile."""
    lhs_cb = ttl.make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = ttl.make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        with lhs_cb.wait() as lhs_tile, rhs_cb.wait() as rhs_tile:
            with out_cb.reserve() as out_tile:
                result = lhs_tile + rhs_tile
                out_tile.store(result)

    @ttl.datamovement()
    def dm_read():
        with lhs_cb.reserve() as lhs_blk, rhs_cb.reserve() as rhs_blk:
            # core(dims=2) returns (x, y) where x=col, y=row
            # tensor indexing is [row, col] = [y, x]
            x, y = ttl.core(dims=2)
            tx_lhs = ttl.copy(lhs[y, x], lhs_blk)
            tx_rhs = ttl.copy(rhs[y, x], rhs_blk)
            tx_lhs.wait()
            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            x, y = ttl.core(dims=2)
            tx = ttl.copy(out_blk, out[y, x])
            tx.wait()

    return ttl.Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


# =============================================================================
# Initial IR Checks - TTL dialect ops with core_x/core_y
# =============================================================================

# CHECK: #ttnn_layout = #ttnn.ttnn_layout<{{.*}}memref<{{.*}}!ttcore.tile<32x32, bf16>{{.*}}>

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: attributes {ttl.base_cta_index = 3 : i32, ttl.crta_indices = [0 : i32, 1 : i32], ttl.kernel_thread = #ttkernel.thread<noc>}

# Verify core_x() and core_y() ops appear in the IR (from x, y = ttl.core(dims=2))
# CHECK: ttl.core_x
# CHECK: ttl.core_y

# CHECK-LABEL: func.func @dm_write
# CHECK: ttl.core_x
# CHECK: ttl.core_y

# =============================================================================
# C++ Kernel Checks - Verify logical coordinates in generated C++
# =============================================================================

# dm_read kernel should use logical coordinates for tile indexing
# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()

# Verify both logical coordinates appear (x before y in generated code)
# CHECK-CPP: get_absolute_logical_x()
# CHECK-CPP: get_absolute_logical_y()

# dm_write kernel should also use logical coordinates
# CHECK-CPP: // dm_write
# CHECK-CPP: void kernel_main()
# CHECK-CPP: get_absolute_logical_x()
# CHECK-CPP: get_absolute_logical_y()


if __name__ == "__main__":
    import torch
    from test_helpers import require_hardware

    print("=== Multicore Add Kernel Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        # 8x8 grid = 256x256 tensor (8 tiles x 8 tiles, one tile per core)
        lhs_torch = torch.full((256, 256), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((256, 256), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((256, 256), dtype=torch.bfloat16)

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

        print("Compiling multicore add kernel...")
        multicore_add(lhs, rhs, out)

        print("=== Multicore Add Kernel Test Complete ===")

    finally:
        ttnn.close_device(device)
