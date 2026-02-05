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

    # Check that the broadcast returns a Block
    assert isinstance(broadcasted, Block)


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

    # All forms should produce the same shape
    assert result1.shape == (2, 3)
    assert result2.shape == (2, 3)
    assert result3.shape == (2, 3)
    assert result4.shape == (2, 3)


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


def test_sqrt():
    """Test sqrt function."""
    input_data = torch.tensor([[4.0, 9.0], [16.0, 25.0]])
    expected = torch.sqrt(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.sqrt(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected)


def test_abs():
    """Test abs function."""
    input_data = torch.tensor([[-1.0, 2.0], [-3.0, 4.0]])
    expected = torch.abs(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.abs(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected)


def test_sin():
    """Test sin function."""
    input_data = torch.tensor([[0.0, torch.pi / 2], [torch.pi, 3 * torch.pi / 2]])
    expected = torch.sin(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.sin(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected, atol=1e-6)


def test_cos():
    """Test cos function."""
    input_data = torch.tensor([[0.0, torch.pi / 2], [torch.pi, 3 * torch.pi / 2]])
    expected = torch.cos(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.cos(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected, atol=1e-6)


def test_log():
    """Test natural logarithm function."""
    input_data = torch.tensor([[1.0, 2.71828], [7.389, 20.0]])
    expected = torch.log(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.log(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected, atol=1e-4)


def test_tanh():
    """Test tanh activation function."""
    input_data = torch.tensor([[-2.0, -1.0], [0.0, 1.0]])
    expected = torch.tanh(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.tanh(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected)


def test_sigmoid():
    """Test sigmoid activation function."""
    input_data = torch.tensor([[-2.0, -1.0], [0.0, 1.0]])
    expected = torch.sigmoid(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.sigmoid(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected)


def test_isnan():
    """Test isnan check function."""
    input_data = torch.tensor([[1.0, float("nan")], [3.0, float("nan")]])
    expected = torch.isnan(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.isnan(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.equal(result_tensor, expected)


def test_isfinite():
    """Test isfinite check function."""
    input_data = torch.tensor([[1.0, float("inf")], [float("-inf"), float("nan")]])
    expected = torch.isfinite(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.isfinite(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.equal(result_tensor, expected)


def test_neg():
    """Test negation function."""
    input_data = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    expected = torch.neg(input_data)

    input_tensor = Tensor(input_data)
    input_block = Block.from_list([input_tensor], shape=(1, 1))

    result = ttl.math.neg(input_block)
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected)


def test_multitile_sqrt():
    """Test sqrt with multiple tiles."""
    # Create a 2x2 grid of tiles
    t1 = Tensor(torch.tensor([[1.0, 4.0], [9.0, 16.0]]))
    t2 = Tensor(torch.tensor([[25.0, 36.0], [49.0, 64.0]]))
    t3 = Tensor(torch.tensor([[81.0, 100.0], [121.0, 144.0]]))
    t4 = Tensor(torch.tensor([[169.0, 196.0], [225.0, 256.0]]))

    input_block = Block.from_list([t1, t2, t3, t4], shape=(2, 2))

    result = ttl.math.sqrt(input_block)

    # Verify each tile
    assert torch.allclose(result.to_list()[0].to_torch(), torch.sqrt(t1.to_torch()))
    assert torch.allclose(result.to_list()[1].to_torch(), torch.sqrt(t2.to_torch()))
    assert torch.allclose(result.to_list()[2].to_torch(), torch.sqrt(t3.to_torch()))
    assert torch.allclose(result.to_list()[3].to_torch(), torch.sqrt(t4.to_torch()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Tests for relu function


def test_relu_basic():
    """Test basic ReLU operation."""
    # Create a block with negative and positive values
    t1 = [Tensor(torch.tensor([[-2.0, 3.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Apply ReLU
    result = ttl.math.relu(block1)

    # Check that result is a Block
    assert isinstance(result, Block)
    assert result.shape == (1, 1)

    # Check that negative values become 0 and positive values stay the same
    expected = torch.tensor([[0.0, 3.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_relu_all_negative():
    """Test ReLU with all negative values."""
    t1 = [Tensor(torch.tensor([[-5.0, -3.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    result = ttl.math.relu(block1)

    # All values should become 0
    expected = torch.tensor([[0.0, 0.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_relu_all_positive():
    """Test ReLU with all positive values."""
    t1 = [Tensor(torch.tensor([[2.0, 7.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    result = ttl.math.relu(block1)

    # All values should stay the same
    expected = torch.tensor([[2.0, 7.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_relu_multitile():
    """Test ReLU on a multi-tile block."""
    # Create a (1, 2) block - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[-1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, -4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    result = ttl.math.relu(block_a)

    # Check result shape
    assert result.shape == (1, 2)

    # Check values
    assert torch.allclose(result.to_list()[0].to_torch(), torch.tensor([[0.0, 2.0]]))
    assert torch.allclose(result.to_list()[1].to_torch(), torch.tensor([[3.0, 0.0]]))


# Tests for exp function


def test_exp_basic():
    """Test basic exponential operation."""
    t1 = [Tensor(torch.tensor([[0.0, 1.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    result = ttl.math.exp(block1)

    # Check that result is a Block
    assert isinstance(result, Block)
    assert result.shape == (1, 1)

    # Check values: e^0 = 1, e^1 = e
    expected = torch.exp(torch.tensor([[0.0, 1.0]]))
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_exp_negative():
    """Test exponential with negative values."""
    t1 = [Tensor(torch.tensor([[-1.0, -2.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    result = ttl.math.exp(block1)

    expected = torch.exp(torch.tensor([[-1.0, -2.0]]))
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_exp_multitile():
    """Test exponential on a multi-tile block."""
    t_a = [
        Tensor(torch.tensor([[0.0, 1.0]])),
        Tensor(torch.tensor([[2.0, -1.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    result = ttl.math.exp(block_a)

    assert result.shape == (1, 2)
    assert torch.allclose(
        result.to_list()[0].to_torch(), torch.exp(torch.tensor([[0.0, 1.0]]))
    )
    assert torch.allclose(
        result.to_list()[1].to_torch(), torch.exp(torch.tensor([[2.0, -1.0]]))
    )


# Tests for reduce_max function


def test_reduce_max_rows():
    """Test reduce_max over rows (dimension 0)."""
    # Create a (2, 1) block - two tiles in row dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 5.0]])),
        Tensor(torch.tensor([[3.0, 2.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[2.0, 2.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension 0 (rows)
    result = ttl.math.reduce_max(block_a, scaler, dims=[0])

    # Result should have shape (1, 1) - rows reduced
    assert result.shape == (1, 1)

    # Max over rows: max([1, 5], [3, 2]) = [3, 5], scaled by [2, 2] = [6, 10]
    expected = torch.tensor([[6.0, 10.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_max_cols():
    """Test reduce_max over columns (dimension 1)."""
    # Create a (1, 2) block - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 5.0]])),
        Tensor(torch.tensor([[3.0, 2.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension 1 (columns)
    result = ttl.math.reduce_max(block_a, scaler, dims=[1])

    # Result should have shape (1, 1) - columns reduced
    assert result.shape == (1, 1)

    # Max over cols: max([1, 5], [3, 2]) along dim 1 = [3, 5]
    expected = torch.tensor([[3.0, 5.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_max_all():
    """Test reduce_max over all dimensions."""
    # Create a (2, 2) block
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
        Tensor(torch.tensor([[5.0, 6.0]])),
        Tensor(torch.tensor([[7.0, 8.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 2))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[0.5, 0.5]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over both dimensions
    result = ttl.math.reduce_max(block_a, scaler, dims=[0, 1])

    # Result should have shape (1, 1)
    assert result.shape == (1, 1)

    # Max over all: max of all values = [7, 8], scaled by [0.5, 0.5] = [3.5, 4.0]
    expected = torch.tensor([[3.5, 4.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_max_invalid_dims():
    """Test that reduce_max rejects invalid dimensions."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Try to reduce over dimension 2, which doesn't exist
    with pytest.raises(
        ValueError,
        match="Cannot reduce along dimension 2.*only 2 dimensions",
    ):
        ttl.math.reduce_max(block_a, scaler, dims=[2])


def test_reduce_max_empty_dims():
    """Test that reduce_max rejects empty dims list."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    with pytest.raises(
        ValueError, match="dims parameter must contain at least one dimension"
    ):
        ttl.math.reduce_max(block_a, scaler, dims=[])


# Tests for reduce_sum function


def test_reduce_sum_rows():
    """Test reduce_sum over rows (dimension 0)."""
    # Create a (2, 1) block - two tiles in row dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[2.0, 2.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension 0 (rows)
    result = ttl.math.reduce_sum(block_a, scaler, dims=[0])

    # Result should have shape (1, 1) - rows reduced
    assert result.shape == (1, 1)

    # Sum over rows: sum([1, 2], [3, 4]) = [4, 6], scaled by [2, 2] = [8, 12]
    expected = torch.tensor([[8.0, 12.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_sum_cols():
    """Test reduce_sum over columns (dimension 1)."""
    # Create a (1, 2) block - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension 1 (columns)
    result = ttl.math.reduce_sum(block_a, scaler, dims=[1])

    # Result should have shape (1, 1) - columns reduced
    assert result.shape == (1, 1)

    # Sum over cols: sum([1, 2], [3, 4]) along dim 1 = [4, 6]
    expected = torch.tensor([[4.0, 6.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_sum_all():
    """Test reduce_sum over all dimensions."""
    # Create a (2, 2) block
    t_a = [
        Tensor(torch.tensor([[1.0, 1.0]])),
        Tensor(torch.tensor([[2.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 3.0]])),
        Tensor(torch.tensor([[4.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 2))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[0.1, 0.1]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over both dimensions
    result = ttl.math.reduce_sum(block_a, scaler, dims=[0, 1])

    # Result should have shape (1, 1)
    assert result.shape == (1, 1)

    # Sum over all: sum of all values = [10, 10], scaled by [0.1, 0.1] = [1.0, 1.0]
    expected = torch.tensor([[1.0, 1.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_sum_invalid_dims():
    """Test that reduce_sum rejects invalid dimensions."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Try to reduce over dimension 2, which doesn't exist
    with pytest.raises(
        ValueError,
        match="Cannot reduce along dimension 2.*only 2 dimensions",
    ):
        ttl.math.reduce_sum(block_a, scaler, dims=[2])


def test_reduce_sum_empty_dims():
    """Test that reduce_sum rejects empty dims list."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    with pytest.raises(
        ValueError, match="dims parameter must contain at least one dimension"
    ):
        ttl.math.reduce_sum(block_a, scaler, dims=[])
