# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Simple element-wise add test for runtime validation."""

from ttlang.d2m_api import *
import torch


@pykernel_gen(
    grid=(1, 1),
    block_factors=[(1, 1), (1, 1), (1, 1)],
    memory_space="L1",
    tiled=True,
)
def add(lhs, rhs, out, block_factors=None, grid=None):
    """Simple element-wise add kernel."""
    lhs_stream = Stream(lhs)
    rhs_stream = Stream(rhs)

    @compute()
    async def add_compute(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        lhs_shard = lhs_cb.pop()
        rhs_shard = rhs_cb.pop()
        out_shard = out_cb.reserve()
        result = lhs_shard + rhs_shard
        out_shard.store(result)
        out_cb.pop()

    @datamovement()
    async def dm_reader(
        lhs_cb: CircularBuffer,
        rhs_cb: CircularBuffer,
        out_cb: CircularBuffer,
    ):
        lhs_shard = lhs_cb.reserve()
        tx = dma(lhs_stream[0, 0], lhs_shard)
        tx.wait()

        rhs_shard = rhs_cb.reserve()
        tx = dma(rhs_stream[0, 0], rhs_shard)
        tx.wait()

        # Write output back to DRAM
        out_shard = out_cb.pop()
        tx = dma(out_shard, lhs_stream[0, 0])  # Reuse lhs_stream layout for output
        tx.wait()

    return Program(add_compute, dm_reader)(lhs, rhs, out)


if __name__ == "__main__":
    print("Creating test tensors...")
    lhs = torch.ones(32, 32, dtype=torch.float32)
    rhs = torch.ones(32, 32, dtype=torch.float32) * 2.0
    out = torch.zeros(32, 32, dtype=torch.float32)

    print("Input lhs:", lhs[0, :5])
    print("Input rhs:", rhs[0, :5])
    print("Input out:", out[0, :5])

    print("\nCompiling and executing kernel...")
    add(lhs, rhs, out)

    print("\nAfter execution:")
    print("Output:", out[0, :5])

    # Verify
    expected = lhs + rhs
    print("Expected:", expected[0, :5])

    # Note: On macOS, runtime execution will be skipped
    # This test is meant to be run on hardware
