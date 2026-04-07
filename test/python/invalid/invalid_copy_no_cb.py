# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: ttl.copy() requires exactly one DFB argument.

This test verifies that calling ttl.copy() with two tensor accessors (no DFB)
raises the expected ValueError.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: error: copy() with tensor subscript dst requires block src
# CHECK-NEXT:   --> {{.*}}invalid_copy_no_cb.py:[[LINE:[0-9]+]]:10
# CHECK-NEXT:    |
# CHECK-NEXT: [[LINE]] |         tx = ttl.copy(lhs[0, 0], rhs[0, 0])
# CHECK-NEXT:    |          ^
# CHECK-NEXT:    |
@ttl.operation(grid=(1, 1))
def invalid_copy_no_dfb_kernel(lhs, rhs, out):
    """This kernel should fail because ttl.copy() needs exactly one DFB."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

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
        # INVALID: ttl.copy() between two tensor accessors (no DFB)
        tx = ttl.copy(lhs[0, 0], rhs[0, 0])
        tx.wait()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


if __name__ == "__main__":
    import torch

    print("=== Copy No DFB Validation Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

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
        invalid_copy_no_dfb_kernel(lhs, rhs, out)

        print("ERROR: Expected ValueError was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
