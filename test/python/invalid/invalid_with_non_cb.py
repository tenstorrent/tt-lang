# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: with statement requires CircularBufferType.

This test verifies that using 'with' on a non-CB value (like a TensorBlock)
raises the expected ValueError.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang import ttl
from ttlang.ttl_api import Program, CircularBuffer, TensorAccessor
from ttlang.operators import copy

import ttnn


# CHECK: Expected CircularBufferType, got
@ttl.kernel(grid=(1, 1))
def invalid_with_non_cb_kernel(lhs, rhs, out):
    """This kernel should fail because 'with' is used on a TensorBlock, not a CB."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @ttl.compute()
    def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        # Get a TensorBlock from wait()
        l = lhs_cb.wait()

        # INVALID: Try to use 'with' pattern on 'l' which is a TensorBlock, not a CB
        # This should fail because 'l.wait()' tries to call wait() on a tensor
        with l.wait() as data:
            r = rhs_cb.wait()
            result = data + r

        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_cb.reserve()
        tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        out_cb.wait()
        tx = copy(out_cb, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


if __name__ == "__main__":
    import torch

    print("=== With Non-CB Validation Test ===")

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
        invalid_with_non_cb_kernel(lhs, rhs, out)

        print("ERROR: Expected ValueError was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
