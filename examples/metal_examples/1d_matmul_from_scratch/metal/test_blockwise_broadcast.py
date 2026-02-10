#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Golden PyTorch block-wise broadcast implementation.

Takes a tensor of shape (block_h, block_w) and repeats it to fill
a larger tensor of shape (H, W) where H is divisible by block_h
and W is divisible by block_w.
"""

import torch


def blockwise_broadcast(block_tensor, H, W):
    """
    Broadcast a block tensor to fill a larger output tensor.

    Args:
        block_tensor: Input tensor of shape (block_h, block_w)
        H: Output height (must be divisible by block_h)
        W: Output width (must be divisible by block_w)

    Returns:
        Output tensor of shape (H, W) with the block repeated
    """
    block_h, block_w = block_tensor.shape

    assert H % block_h == 0, f"H ({H}) must be divisible by block_h ({block_h})"
    assert W % block_w == 0, f"W ({W}) must be divisible by block_w ({block_w})"

    num_blocks_y = H // block_h
    num_blocks_x = W // block_w

    # Method 1: Using repeat
    # Repeat along each dimension
    output = block_tensor.repeat(num_blocks_y, num_blocks_x)

    return output


def blockwise_broadcast_tile(block_tensor, H, W):
    """
    Alternative implementation using tile (similar to repeat but different semantics).

    Args:
        block_tensor: Input tensor of shape (block_h, block_w)
        H: Output height (must be divisible by block_h)
        W: Output width (must be divisible by block_w)

    Returns:
        Output tensor of shape (H, W) with the block repeated
    """
    block_h, block_w = block_tensor.shape

    assert H % block_h == 0, f"H ({H}) must be divisible by block_h ({block_h})"
    assert W % block_w == 0, f"W ({W}) must be divisible by block_w ({block_w})"

    num_blocks_y = H // block_h
    num_blocks_x = W // block_w

    # Add dimensions and tile
    output = block_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, block_h, block_w)
    output = output.tile(
        num_blocks_y, num_blocks_x, 1, 1
    )  # (num_blocks_y, num_blocks_x, block_h, block_w)
    output = output.permute(
        0, 2, 1, 3
    )  # (num_blocks_y, block_h, num_blocks_x, block_w)
    output = output.reshape(H, W)  # (H, W)

    return output


def blockwise_broadcast_expand(block_tensor, H, W):
    """
    Alternative implementation using expand (memory-efficient, no copy).

    Args:
        block_tensor: Input tensor of shape (block_h, block_w)
        H: Output height (must be divisible by block_h)
        W: Output width (must be divisible by block_w)

    Returns:
        Output tensor of shape (H, W) with the block repeated (view, not copy)
    """
    block_h, block_w = block_tensor.shape

    assert H % block_h == 0, f"H ({H}) must be divisible by block_h ({block_h})"
    assert W % block_w == 0, f"W ({W}) must be divisible by block_w ({block_w})"

    num_blocks_y = H // block_h
    num_blocks_x = W // block_w

    # Reshape and expand
    output = block_tensor.unsqueeze(0).unsqueeze(2)  # (1, block_h, 1, block_w)
    output = output.expand(num_blocks_y, block_h, num_blocks_x, block_w)
    print(output)
    output = output.reshape(H, W)

    # Note: This returns a view, use .contiguous() if you need a copy
    return output.contiguous()


if __name__ == "__main__":
    # Test with different block sizes
    block_h, block_w = 2, 3
    H, W = 6, 9

    # Create a simple block tensor with sequential values
    block = torch.arange(block_h * block_w, dtype=torch.float32).reshape(
        block_h, block_w
    )
    print(f"Input block ({block_h}x{block_w}):")
    print(block)
    print()

    # Test method 1: repeat
    output1 = blockwise_broadcast(block, H, W)
    print(f"Output using repeat ({H}x{W}):")
    print(output1)
    print()

    # Test method 2: tile
    output2 = blockwise_broadcast_tile(block, H, W)
    print(f"Output using tile ({H}x{W}):")
    print(output2)
    print()

    # Test method 3: expand
    output3 = blockwise_broadcast_expand(block, H, W)
    print(f"Output using expand ({H}x{W}):")
    print(output3)
    print()

    # Verify all methods produce the same result
    assert torch.allclose(output1, output2), "repeat and tile methods differ!"
    assert torch.allclose(output1, output3), "repeat and expand methods differ!"
    print("✓ All methods produce identical results!")

    # Test with different dimensions
    print("\n" + "=" * 50)
    print("Testing with block_h=4, block_w=8, H=12, W=16")
    block_h, block_w = 4, 8
    H, W = 12, 16
    block = torch.randn(block_h, block_w)
    output = blockwise_broadcast(block, H, W)
    print(f"Output shape: {output.shape}")

    # Verify the pattern
    num_blocks_y = H // block_h
    num_blocks_x = W // block_w
    print(f"Number of blocks: {num_blocks_y} x {num_blocks_x}")

    # Check that each block is identical to the original
    all_match = True
    for by in range(num_blocks_y):
        for bx in range(num_blocks_x):
            block_slice = output[
                by * block_h : (by + 1) * block_h, bx * block_w : (bx + 1) * block_w
            ]
            if not torch.allclose(block_slice, block):
                all_match = False
                print(f"Block at ({by}, {bx}) doesn't match!")

    if all_match:
        print("✓ All blocks match the original!")
