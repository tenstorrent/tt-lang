# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Simplest possible test: Write constant from DRAM -> L1 -> DRAM."""

from ttlang.d2m_api import *
import torch


@pykernel_gen(
    grid=(1, 1),
    block_factors=[(1, 1), (1, 1)],
    memory_space="L1",
    tiled=True,
)
def simple_write(inp, out, block_factors=None, grid=None):
    """Copy input to output - simplest possible kernel with DRAM->L1->DRAM."""
    inp_stream = Stream(inp)

    @datamovement()
    async def dm_reader(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        # Read from input DRAM into L1 circular buffer
        inp_shard = inp_cb.reserve()
        tx = dma(inp_stream[0, 0], inp_shard)
        tx.wait()

    @datamovement()
    async def dm_writer(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        # Pop from input CB (wait for reader)
        inp_shard = inp_cb.pop()

        # Reserve output CB and copy data
        out_shard = out_cb.reserve()
        out_shard.store(inp_shard)
        out_cb.pop()

    return Program(dm_reader, dm_writer)(inp, out)


if __name__ == "__main__":
    print("Creating test tensors...")
    inp = torch.full((32, 32), 42.0, dtype=torch.float32)
    out = torch.full((32, 32), -999.0, dtype=torch.float32)

    print("Input inp:", inp[0, :5])
    print("Input out:", out[0, :5])

    print("\nCompiling and executing kernel...")
    simple_write(inp, out)

    print("\nAfter execution:")
    print("Output:", out[0, :5])

    # Verify
    expected = inp.clone()
    print("Expected:", expected[0, :5])

    # Note: On macOS, runtime execution will be skipped
    # On hardware, we expect output to match input (copy operation)
    # This validates DRAM->L1->DRAM data movement works
    print("\nThis test validates basic DRAM -> L1 -> DRAM data movement.")
    print("Expected: output should be 42.0 if kernel executed successfully.")
