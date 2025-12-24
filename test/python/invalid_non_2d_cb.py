# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %run-test %s 2>&1 | FileCheck %s

"""
Validation test: only 2D CBs are supported.

This test verifies that using a 3D tensor (which creates 3D CB) raises ValueError.
The CB shape validation check comes before tensor validation in the code path.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang.ttl_api import (
    pykernel_gen,
    Program,
    CircularBuffer,
    TensorAccessor,
    compute,
    datamovement,
)
from ttlang.operators import copy

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


# CHECK: ValueError: Only 2D CBs supported, got shape
@pykernel_gen(grid=(1, 1), block_factors=[(1, 1, 1), (1, 1, 1), (1, 1, 1)])
def invalid_3d_cb_kernel(lhs, rhs, out):
    """This kernel should fail because 3D tensors create 3D CBs which are not supported."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @compute()
    def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_read(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_cb.reserve()
        tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @datamovement()
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

    print("=== Non-2D CB Validation Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        # Create 3D tensors (batch x height x width)
        lhs_torch = torch.full((1, 32, 32), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((1, 32, 32), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((1, 32, 32), dtype=torch.bfloat16)

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
        invalid_3d_cb_kernel(lhs, rhs, out)

        print("ERROR: Expected ValueError was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
