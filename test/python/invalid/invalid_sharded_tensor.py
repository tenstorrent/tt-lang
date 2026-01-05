# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: TTNN interop requires interleaved tensors.

This test verifies that using a sharded (non-interleaved) tensor
raises the expected ValueError.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program


# CHECK: ValueError: TTNN interop requires interleaved tensors
@ttl.kernel(grid=(1, 1))
def invalid_sharded_kernel(lhs, rhs, out):
    """This kernel should fail because sharded tensors are not supported."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_cb = make_circular_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_cb.reserve()
        tx_lhs = copy(lhs[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


if __name__ == "__main__":
    import torch

    print("=== Sharded Tensor Validation Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        rhs_torch = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Create sharded memory config (non-interleaved)
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
            ),
            (32, 32),  # shard shape
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

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

        # Convert to sharded memory layout (non-interleaved)
        lhs = ttnn.to_memory_config(lhs, memory_config=sharded_mem_config)
        rhs = ttnn.to_memory_config(rhs, memory_config=sharded_mem_config)
        out = ttnn.to_memory_config(out, memory_config=sharded_mem_config)

        # This should raise ValueError
        invalid_sharded_kernel(lhs, rhs, out)

        print("ERROR: Expected ValueError was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
