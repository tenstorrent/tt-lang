#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Example demonstrating ttl.math.broadcast function.

This example matches the specification's element-wise with broadcast example:
Y = sqrt(A^2 + B^2) where A has shape (1, N) and B has shape (1, 1).
"""

import os
import sys

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from sim import ttl, ttnn


@ttl.kernel(grid="auto")
def elementwise_with_broadcast(
    A: ttnn.Tensor,
    B: ttnn.Tensor,
    Y: ttnn.Tensor,
) -> None:
    """Element-wise operation with broadcast: Y = sqrt(A^2 + B^2).

    Tensor shapes:
        A: (1, N) -> (1, NT) in tiles
        B: (1, 1) -> (1, 1) in tiles
        Y: (1, N) -> (1, NT) in tiles

    Where NT = N // TILE_SIZE
    """
    # Calculate tile counts
    N = A.shape[1]
    TILE_SIZE = ttl.TILE_SHAPE[1]
    NT = N // TILE_SIZE

    # Create circular buffers
    a_cb = ttl.make_circular_buffer_like(A, shape=(1, 1))
    b_cb = ttl.make_circular_buffer_like(B, shape=(1, 1))
    y_cb = ttl.make_circular_buffer_like(Y, shape=(1, 1))

    @ttl.datamovement()
    def elwise_read():
        for nt in range(NT):
            # Acquire a_blk and b_blk from a_cb and b_cb
            with a_cb.reserve() as a_blk, b_cb.reserve() as b_blk:
                # Copy data using slice notation
                a_xf = ttl.copy(A[0, nt], a_blk)
                b_xf = ttl.copy(B[0, 0], b_blk)

                a_xf.wait()
                b_xf.wait()

                # Release a_blk and b_blk

    @ttl.compute()
    def elwise_compute():
        for _ in range(NT):
            # Acquire a_blk, b_blk and y_blk from a_cb, b_cb and y_cb
            with (
                a_cb.wait() as a_blk,
                b_cb.wait() as b_blk,
                y_cb.reserve() as y_blk,
            ):
                # Compute y = sqrt(a^2 + b^2)
                # Note: Using broadcast to expand B from (1,1) to match A's (1,NT)
                a_squared = a_blk**2
                b_squared = b_blk**2

                # Broadcast b_squared along dimension 1 (columns) to match a_squared
                y = a_squared + ttl.math.broadcast(b_squared, dims=[1])
                y_blk.store(y)

                # Release a_blk, b_blk and y_blk

    @ttl.datamovement()
    def elwise_write():
        for nt in range(NT):
            # Acquire y_blk from y_cb
            with y_cb.wait() as y_blk:
                # Copy result using slice notation
                y_xf = ttl.copy(y_blk, Y[0, nt])
                y_xf.wait()

                # Release y_blk


def main():
    """Run the broadcast example."""
    print("=" * 60)
    print("TT-Lang ttl.math.broadcast Example")
    print("=" * 60)
    print()

    # Create test tensors
    # A: shape (1, 128) = (1, 4) in tiles
    # B: shape (1, 32) = (1, 1) in tiles (we'll treat it as broadcast source)
    TILE_SIZE = 32
    N = 128
    NT = N // TILE_SIZE

    print(f"Tensor A shape: (1, {N}) = (1, {NT}) in tiles")
    print(f"Tensor B shape: (1, {TILE_SIZE}) = (1, 1) in tiles")
    print(f"Tensor Y shape: (1, {N}) = (1, {NT}) in tiles")
    print()

    # Create tensors with specific values for easy verification
    A = ttnn.from_torch(torch.full((1, N), 3.0, dtype=torch.float32))
    B = ttnn.from_torch(torch.full((1, TILE_SIZE), 4.0, dtype=torch.float32))
    Y = ttnn.empty((1, N), dtype=torch.float32)

    print("Running kernel...")
    elementwise_with_broadcast(A, B, Y)

    # Verify result
    # Expected: Y = A^2 + B^2 = 3^2 + 4^2 = 9 + 16 = 25
    Y_torch = Y.to_torch()
    expected = torch.full((1, N), 25.0, dtype=torch.float32)

    print()
    print("Results:")
    print(f"  Y sample values: {Y_torch[0, :5].tolist()}")
    print(f"  Expected: {expected[0, :5].tolist()}")
    print()

    if torch.allclose(Y_torch, expected):
        print("✓ Result matches expected output!")
        print(f"✓ Broadcast successfully expanded B from (1,1) to (1,{NT}) tiles")
        return 0
    else:
        print("✗ Result does not match expected output")
        return 1


if __name__ == "__main__":
    sys.exit(main())
