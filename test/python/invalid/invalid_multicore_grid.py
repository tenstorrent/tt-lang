# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: core(dims=N) only supports dims=2.

This test verifies that using dims=3 with core() raises ValueError.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: core() currently only supports dims=2, got dims=3
@ttl.kernel(grid=(2, 2))
def invalid_core_dims_kernel(lhs, rhs, out):
    """This kernel should fail because core(dims=3) is not supported."""
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
            # This should fail - dims=3 not supported
            x, y, z = ttl.core(dims=3)
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


if __name__ == "__main__":
    import torch

    print("=== Invalid core(dims=3) Validation Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((64, 64), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((64, 64), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

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

        # This should raise ValueError
        invalid_core_dims_kernel(lhs, rhs, out)

        print("ERROR: Expected ValueError was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
