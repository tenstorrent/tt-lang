#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Example: Using wait()/reserve() with context managers

Demonstrates that you can call wait() or reserve() to get a block,
then later use it with a 'with' statement - the context manager
automatically handles push()/pop() on exit.
"""

import torch
from python.sim.dfb import DataflowBuffer, Block
from python.sim.ttnnsim import Tensor
from python.sim.kernel import kernel
from python.sim.copy import copy
from python.sim import ttl


def make_ones_tile():
    return Tensor(torch.ones(32, 32))


def make_zeros_tile():
    return Tensor(torch.zeros(32, 32))


@kernel(grid=(1,))
def example_reserve_then_with():
    """Example: x = dfb.reserve(); ...; with x: ..."""
    print("\n=== Example 1: reserve() → with statement ===")

    element = make_ones_tile()
    data = make_ones_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    result_tensor = make_zeros_tile()

    @ttl.compute()
    def compute_thread():
        # Reserve block explicitly
        x = dfb.reserve()
        print(f"✓ Reserved block: {type(x).__name__}")

        # Do some work with it
        x.store(Block.from_tensor(data))
        print("✓ Stored data in block")

        # Later, use 'with' statement
        # The 'with' will call push() automatically on exit
        with x:
            print("✓ Inside 'with' - about to exit")

        print("✓ Exited 'with' - push() called automatically!")

        # Verify push() worked - wait for the data
        y = dfb.wait()
        with y:
            out_block = out_dfb.reserve()
            out_block.store(y)
            out_block.push()

    @ttl.datamovement()
    def dm1():
        pass

    @ttl.datamovement()
    def dm2():
        # Use context manager with copy for DM thread
        with out_dfb.wait() as result:
            tx = copy(result, result_tensor)
            tx.wait()


@kernel(grid=(1,))
def example_wait_then_with():
    """Example: x = dfb.wait(); ...; with x: ..."""
    print("\n=== Example 2: wait() → with statement ===")

    element = make_ones_tile()
    data = make_ones_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    result_tensor = make_zeros_tile()

    @ttl.compute()
    def compute_thread():
        # First produce some data
        block = dfb.reserve()
        block.store(Block.from_tensor(data))
        block.push()

        # Now wait() to get a block
        x = dfb.wait()
        print(f"✓ Waited for block: {type(x).__name__}")

        # Later, use 'with' statement
        # The 'with' will call pop() automatically on exit
        with x:
            # Use the block - must use as STORE_SRC before pop in COMPUTE thread
            out_block = out_dfb.reserve()
            out_block.store(x)
            out_block.push()
            print("✓ Inside 'with' - using block")

        print("✓ Exited 'with' - pop() called automatically!")

    @ttl.datamovement()
    def dm1():
        pass

    @ttl.datamovement()
    def dm2():
        # Use context manager with copy for DM thread
        with out_dfb.wait() as result:
            tx = copy(result, result_tensor)
            tx.wait()


if __name__ == "__main__":
    print("Examples of using wait()/reserve() with context managers\n")
    print("Key insight: You can get a block, do stuff with it, then")
    print("later use 'with block:' to automatically push()/pop() on exit.")

    example_reserve_then_with()
    example_wait_then_with()

    print("\n✅ All examples completed successfully!")
