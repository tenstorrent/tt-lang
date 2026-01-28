# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for ttl.math module functions.

Includes tests for ttl.math.broadcast and verification of explicit broadcasting requirements.
"""

import pytest
import torch

from sim import ttl
from sim.block import Block
from sim.ttnnsim import Tensor


def test_broadcast_basic():
    """Test basic broadcast operation."""
    # Create a (1, 1) block - single tile
    t1 = [Tensor(torch.tensor([[5.0, 6.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast along dimension 1 (columns)
    broadcasted = ttl.math.broadcast(block1, dims=[1])

    # Check that broadcast returns a Block
    assert isinstance(broadcasted, Block)

    # The shape should still be (1, 1) - actual broadcasting happens during operations
    assert broadcasted.shape == (1, 1)


def test_broadcast_with_operation():
    """Test broadcast in the context of an operation."""
    # Create blocks of different shapes
    # Block A: (1, 2) - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Block B: (1, 1) - single tile
    t_b = [Tensor(torch.tensor([[10.0, 20.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Broadcast B and add to A
    # This simulates: A + broadcast(B, dims=[1])
    broadcasted_b = ttl.math.broadcast(block_b, dims=[1])

    # The addition should use broadcasting
    result = block_a + broadcasted_b

    # Result should be a Block with shape (1, 2)
    assert isinstance(result, Block)
    assert result.shape == (1, 2)

    # Check values - B should have been broadcast to both tiles of A
    for i in range(2):
        result_tensor = result[i].to_torch()
        if i == 0:
            expected = torch.tensor([[11.0, 22.0]])  # [1, 2] + [10, 20]
        else:
            expected = torch.tensor([[13.0, 24.0]])  # [3, 4] + [10, 20]
        assert torch.allclose(result_tensor, expected)


def test_broadcast_example_from_spec():
    """Test the broadcast example from the specification.

    From spec: y = ttl.math.sqrt(a_squared + ttl.math.broadcast(b_squared, dims=[1]))
    Where a_squared has shape (1, N) and b_squared has shape (1, 1)
    """
    # Create a_squared with shape (1, 3)
    t_a = [
        Tensor(torch.tensor([[9.0, 16.0]])),
        Tensor(torch.tensor([[25.0, 36.0]])),
        Tensor(torch.tensor([[49.0, 64.0]])),
    ]
    a_squared = Block.from_list(t_a, shape=(1, 3))

    # Create b_squared with shape (1, 1)
    t_b = [Tensor(torch.tensor([[16.0, 16.0]]))]
    b_squared = Block.from_list(t_b, shape=(1, 1))

    # Broadcast b_squared along dimension 1
    b_broadcast = ttl.math.broadcast(b_squared, dims=[1])

    # Add them together (should broadcast b to match a's shape)
    result = a_squared + b_broadcast

    # Check result shape
    assert result.shape == (1, 3)

    # Check values - b should have been added to all tiles of a
    expected_values = [
        torch.tensor([[25.0, 32.0]]),  # [9, 16] + [16, 16]
        torch.tensor([[41.0, 52.0]]),  # [25, 36] + [16, 16]
        torch.tensor([[65.0, 80.0]]),  # [49, 64] + [16, 16]
    ]

    for i in range(3):
        result_tensor = result[i].to_torch()
        assert torch.allclose(result_tensor, expected_values[i])


def test_broadcast_multiple_dims():
    """Test broadcast along multiple dimensions."""
    # Create a (1, 1) block
    t1 = [Tensor(torch.tensor([[2.0, 3.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast along both dimensions
    broadcasted = ttl.math.broadcast(block1, dims=[0, 1])

    # Check that it returns a Block
    assert isinstance(broadcasted, Block)


def test_broadcast_preserves_data():
    """Test that broadcast preserves the original data."""
    # Create a block with specific values
    original_value = torch.tensor([[7.0, 8.0]])
    t1 = [Tensor(original_value.clone())]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast it
    broadcasted = ttl.math.broadcast(block1, dims=[1])

    # Check that the data is preserved
    broadcasted_tensor = broadcasted[0].to_torch()
    assert torch.allclose(broadcasted_tensor, original_value)


# Tests for explicit broadcasting requirements


def test_implicit_broadcast_rejected():
    """Test that implicit broadcasting without ttl.math.broadcast() raises an error."""
    # Create blocks of different shapes
    # Block A: (1, 2) - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Block B: (1, 1) - single tile (would need broadcasting)
    t_b = [Tensor(torch.tensor([[10.0, 20.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Trying to add without explicit broadcast should raise ValueError
    with pytest.raises(
        ValueError,
        match="Cannot perform operation: shape mismatch.*Use ttl.math.broadcast",
    ):
        result = block_a + block_b


def test_implicit_broadcast_different_shapes():
    """Test that operations with mismatched shapes are rejected."""
    # Block A: (2, 1)
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Block B: (1, 2)
    t_b = [
        Tensor(torch.tensor([[5.0, 6.0]])),
        Tensor(torch.tensor([[7.0, 8.0]])),
    ]
    block_b = Block.from_list(t_b, shape=(1, 2))

    # Trying to multiply without explicit broadcast should raise ValueError
    with pytest.raises(
        ValueError,
        match="Cannot perform operation: shape mismatch.*Use ttl.math.broadcast",
    ):
        result = block_a * block_b


def test_matching_shapes_allowed():
    """Test that operations with matching shapes work without broadcast."""
    # Both blocks have shape (1, 2)
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    t_b = [
        Tensor(torch.tensor([[10.0, 20.0]])),
        Tensor(torch.tensor([[30.0, 40.0]])),
    ]
    block_b = Block.from_list(t_b, shape=(1, 2))

    # This should work - shapes match exactly
    result = block_a + block_b
    assert isinstance(result, Block)
    assert result.shape == (1, 2)


def test_broadcast_on_wrong_dimension_rejected():
    """Test that broadcasting on a dimension with size != 1 is rejected."""
    # Block with shape (2, 1) - cannot broadcast on dimension 0
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Try to broadcast on dimension 0, which has size 2
    with pytest.raises(
        ValueError,
        match="Cannot broadcast along dimension 0: dimension must have size 1",
    ):
        ttl.math.broadcast(block_a, dims=[0])


def test_broadcast_out_of_range_rejected():
    """Test that broadcasting on non-existent dimension is rejected."""
    # Block with shape (1, 1) - only has dimensions 0 and 1
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    # Try to broadcast on dimension 2, which doesn't exist
    with pytest.raises(
        ValueError,
        match="Cannot broadcast along dimension 2.*only 2 dimensions",
    ):
        ttl.math.broadcast(block_a, dims=[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
