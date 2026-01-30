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
    """Test that implicit broadcasting works automatically when shapes are compatible."""
    # Create blocks of different shapes
    # Block A: (1, 2) - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Block B: (1, 1) - single tile (will be broadcast automatically)
    t_b = [Tensor(torch.tensor([[10.0, 20.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Implicit broadcasting should work automatically
    result = block_a + block_b

    # Result should have shape (1, 2) and correct values
    assert result._shape == (1, 2)
    # First tile: [1, 2] + [10, 20] = [11, 22]
    assert torch.allclose(result[0]._tensor, torch.tensor([[11.0, 22.0]]))
    # Second tile: [3, 4] + [10, 20] = [13, 24]
    assert torch.allclose(result[1]._tensor, torch.tensor([[13.0, 24.0]]))


def test_implicit_broadcast_different_shapes():
    """Test that implicit broadcasting works with different compatible shapes."""
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

    # Implicit broadcasting should work automatically (both dimensions have a 1)
    result = block_a * block_b

    # Result should have shape (2, 2) - both dimensions broadcast
    assert result._shape == (2, 2)
    # Verify the result has 4 tiles (2x2)
    assert len(result) == 4


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


# Tests for all different forms of broadcast usage


def test_all_broadcast_forms():
    """Test all different forms of broadcast usage work correctly.

    Tests the four forms:
    1) result = a * b - direct implicit broadcast
    2) result = a * broadcast(b, dims=[1]) - explicit broadcast with dims
    3) result = a * broadcast(b, y_unused, dims=[1]) - explicit with unused output hint
    4) w = broadcast(b, dims=[1]); result = a * w - intermediate variable

    Note: We can't test with .store() on temporary blocks created via from_list(),
    as those are already in DONE state. The patterns above are what work in real code
    with circular buffers.
    """
    # Setup: 'a' is MxN (2x3) and 'b' is Mx1 (2x1)
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),  # Row 0, Col 0
        Tensor(torch.tensor([[3.0, 4.0]])),  # Row 0, Col 1
        Tensor(torch.tensor([[5.0, 6.0]])),  # Row 0, Col 2
        Tensor(torch.tensor([[7.0, 8.0]])),  # Row 1, Col 0
        Tensor(torch.tensor([[9.0, 10.0]])),  # Row 1, Col 1
        Tensor(torch.tensor([[11.0, 12.0]])),  # Row 1, Col 2
    ]
    block_a = Block.from_list(t_a, shape=(2, 3))

    t_b = [
        Tensor(torch.tensor([[2.0, 2.0]])),  # Row 0, Col 0
        Tensor(torch.tensor([[3.0, 3.0]])),  # Row 1, Col 0
    ]
    block_b = Block.from_list(t_b, shape=(2, 1))

    # Form 1: Direct implicit broadcast
    result1 = block_a * block_b

    # Form 2: Explicit broadcast with dims
    result2 = block_a * ttl.math.broadcast(block_b, dims=[1])

    # Form 3: Explicit broadcast with unused output hint (None since we can't create a CB here)
    result3 = block_a * ttl.math.broadcast(block_b, None, dims=[1])

    # Form 4: Store broadcast result first, then use it
    broadcast_b = ttl.math.broadcast(block_b, dims=[1])
    result4 = block_a * broadcast_b

    # All forms should produce the same result
    # Expected: each element of a multiplied by corresponding broadcast element of b
    # Row 0: [1,2] * [2,2] = [2,4], [3,4] * [2,2] = [6,8], [5,6] * [2,2] = [10,12]
    # Row 1: [7,8] * [3,3] = [21,24], [9,10] * [3,3] = [27,30], [11,12] * [3,3] = [33,36]
    expected = [
        torch.tensor([[2.0, 4.0]]),
        torch.tensor([[6.0, 8.0]]),
        torch.tensor([[10.0, 12.0]]),
        torch.tensor([[21.0, 24.0]]),
        torch.tensor([[27.0, 30.0]]),
        torch.tensor([[33.0, 36.0]]),
    ]

    # Verify all forms produce the same correct result
    for i in range(6):
        assert torch.allclose(
            result1[i]._tensor, expected[i]
        ), f"Form 1 tile {i} mismatch"
        assert torch.allclose(
            result2[i]._tensor, expected[i]
        ), f"Form 2 tile {i} mismatch"
        assert torch.allclose(
            result3[i]._tensor, expected[i]
        ), f"Form 3 tile {i} mismatch"
        assert torch.allclose(
            result4[i]._tensor, expected[i]
        ), f"Form 4 tile {i} mismatch"


def test_broadcast_form1_direct_implicit():
    """Test form 1: y.store(a * b) with direct implicit broadcast."""
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
        Tensor(torch.tensor([[5.0, 6.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0, 10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Direct broadcast: no ttl.math.broadcast() call needed
    result = block_a * block_b

    assert result.shape == (1, 3)
    assert torch.allclose(result[0]._tensor, torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(result[1]._tensor, torch.tensor([[30.0, 40.0]]))
    assert torch.allclose(result[2]._tensor, torch.tensor([[50.0, 60.0]]))


def test_broadcast_form2_explicit_dims():
    """Test form 2: y.store(a * broadcast(b, dims=[1]))."""
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
        Tensor(torch.tensor([[5.0, 6.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0, 10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Explicit broadcast with dims parameter
    result = block_a * ttl.math.broadcast(block_b, dims=[1])

    assert result.shape == (1, 3)
    assert torch.allclose(result[0]._tensor, torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(result[1]._tensor, torch.tensor([[30.0, 40.0]]))
    assert torch.allclose(result[2]._tensor, torch.tensor([[50.0, 60.0]]))


def test_broadcast_form3_with_output_hint():
    """Test form 3: y.store(a * broadcast(b, y, dims=[1]))."""
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
        Tensor(torch.tensor([[5.0, 6.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0, 10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    t_y = [Tensor(torch.zeros(1, 2)) for _ in range(3)]
    block_y = Block.from_list(t_y, shape=(1, 3))

    # Explicit broadcast with output block hint (unused but accepted)
    result = block_a * ttl.math.broadcast(block_b, block_y, dims=[1])

    assert result.shape == (1, 3)
    assert torch.allclose(result[0]._tensor, torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(result[1]._tensor, torch.tensor([[30.0, 40.0]]))
    assert torch.allclose(result[2]._tensor, torch.tensor([[50.0, 60.0]]))


def test_broadcast_form4_intermediate_store():
    """Test form 4: w = broadcast(b, dims=[1]); result = a * w."""
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
        Tensor(torch.tensor([[5.0, 6.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0, 10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Store broadcast result in w first (as an intermediate variable, not .store())
    broadcast_b = ttl.math.broadcast(block_b, dims=[1])

    # Then use it in the operation
    result = block_a * broadcast_b

    assert result.shape == (1, 3)
    assert torch.allclose(result[0]._tensor, torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(result[1]._tensor, torch.tensor([[30.0, 40.0]]))
    assert torch.allclose(result[2]._tensor, torch.tensor([[50.0, 60.0]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
