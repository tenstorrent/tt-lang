# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# type: ignore

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_MIXED_DTYPE_CASE=add not %python %s 2>&1 | FileCheck %s --check-prefix=ADD
# RUN: env TTLANG_MIXED_DTYPE_CASE=store not %python %s 2>&1 | FileCheck %s --check-prefix=STORE
# RUN: env TTLANG_MIXED_DTYPE_CASE=copy not %python %s 2>&1 | FileCheck %s --check-prefix=COPY
# RUN: env TTLANG_MIXED_DTYPE_CASE=matmul not %python %s 2>&1 | FileCheck %s --check-prefix=MATMUL

"""
Validation test: operations that combine tile values require matching data types.

Trying to combine DFB blocks or tensor slices with different element types should
produce a dtype-specific diagnostic rather than a shape or broadcast diagnostic.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl

from ttlang_test_utils import to_l1


# ADD: incompatible tensor data types for operation: got (1, 1) f32 tensor and (1, 1) bf16 tensor; operation requires matching data types
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


# STORE: incompatible tensor data types for store: got (1, 1) f32 tensor and (1, 1) bf16 tensor; store requires matching data types
@ttl.operation(grid=(1, 1))
def mixed_dtype_store_kernel(lhs, rhs, out):
    """INVALID: store an f32 tile into a bf16 reserve block."""
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with rhs_dfb.wait() as rhs_tile:
            with out_dfb.reserve() as out_tile:
                out_tile.store(rhs_tile)

    @ttl.datamovement()
    def dm_read():
        with rhs_dfb.reserve() as rhs_blk:
            tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            tx_out = ttl.copy(out_blk, out[0, 0])
            tx_out.wait()


# COPY: incompatible tensor data types for copy: got (1, 1) f32 tensor and (1, 1) bf16 tensor; copy requires matching data types
@ttl.operation(grid=(1, 1))
def mixed_dtype_copy_kernel(lhs, rhs, out):
    """INVALID: copy an f32 tensor tile into a bf16 DFB."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with lhs_dfb.wait() as lhs_tile:
            with out_dfb.reserve() as out_tile:
                out_tile.store(lhs_tile)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as lhs_blk:
            tx_rhs = ttl.copy(rhs[0, 0], lhs_blk)
            tx_rhs.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            tx_out = ttl.copy(out_blk, out[0, 0])
            tx_out.wait()


# MATMUL: incompatible tensor data types for matmul: got (1, 1) bf16 tensor and (1, 1) f32 tensor; matmul requires matching data types
@ttl.operation(grid=(1, 1))
def mixed_dtype_matmul_kernel(lhs, rhs, out):
    """INVALID: matmul bf16 and f32 tiles."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with lhs_dfb.wait() as lhs_tile, rhs_dfb.wait() as rhs_tile:
            with out_dfb.reserve() as out_tile:
                out_tile.store(lhs_tile @ rhs_tile)

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

        case = os.environ["TTLANG_MIXED_DTYPE_CASE"]
        if case == "add":
            mixed_dtype_add_kernel(lhs, rhs, out)
        elif case == "store":
            out_bf16 = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)
            mixed_dtype_store_kernel(lhs, rhs, out_bf16)
        elif case == "copy":
            mixed_dtype_copy_kernel(lhs, rhs, out)
        elif case == "matmul":
            mixed_dtype_matmul_kernel(lhs, rhs, out)
        else:
            raise ValueError(f"unknown test case: {case}")

        print("ERROR: Expected TypeError was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
