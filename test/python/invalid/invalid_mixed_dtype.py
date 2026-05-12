# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: frontend binary ops require matching data types.

Trying to add DFB blocks with different element types should produce a
type diagnostic from the TTL dialect rather than a shape or broadcast
diagnostic.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl

from ttlang_test_utils import to_l1


# CHECK: incompatible tensor data types
# CHECK-SAME: requires matching data types
@ttl.operation(grid=(1, 1))
def mixed_dtype_add_kernel(lhs, rhs, out):
    """INVALID: add same-shaped tiles with different data types."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with lhs_dfb.wait() as lhs_tile, rhs_dfb.wait() as rhs_tile:
            with out_dfb.reserve() as out_tile:
                out_tile.store(lhs_tile + rhs_tile)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_blk:
            tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
            tx_lhs.wait()
        with rhs_dfb.reserve() as rhs_blk:
            tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            tx_out = ttl.copy(out_blk, out[0, 0])
            tx_out.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        lhs = to_l1(torch.ones((32, 32), dtype=torch.bfloat16), device)
        rhs = to_l1(torch.ones((32, 32), dtype=torch.float32), device)
        out = to_l1(torch.zeros((32, 32), dtype=torch.float32), device)

        mixed_dtype_add_kernel(lhs, rhs, out)

        print("ERROR: Expected TypeError was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
