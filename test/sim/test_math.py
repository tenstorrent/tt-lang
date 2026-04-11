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
from sim.dfb import Block
from sim.ttnnsim import Tensor


def test_broadcast_basic():
    """Test basic broadcast operation."""
    # Create a (1, 1) block with element_shape=(1, 1) for broadcasting along dim 0
    t1 = [Tensor(torch.tensor([[5.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast along dimension 0 (outermost/rows); (1,1) has both dims = 1
    broadcasted = ttl.math.broadcast(block1, dims=[0])

    # Check that broadcast returns a Block
    assert isinstance(broadcasted, Block)

    # The shape should still be (1, 1) - actual broadcasting happens during operations
    assert broadcasted.shape == (1, 1)


def test_broadcast_with_operation():
    """Test broadcast in the context of an operation."""
    # Create blocks of different shapes
    # Block A: (1, 2) - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Block B: (1, 1) - single tile with element_shape=(1,1) for broadcasting
    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Broadcast B and add to A
    # This simulates: A + broadcast(B, dims=[0])
    broadcasted_b = ttl.math.broadcast(block_b, dims=[0])

    # The addition should use broadcasting
    result = block_a + broadcasted_b

    # Result should be a Block with shape (1, 2)
    assert isinstance(result, Block)
    assert result.shape == (1, 2)


def test_broadcast_example_from_spec():
    """Test the broadcast example from the specification.

    From spec: y = ttl.math.sqrt(a_squared + ttl.math.broadcast(b_squared, dims=[0]))
    Where a_squared has shape (1, N) and b_squared has shape (1, 1)
    """
    # Create a_squared with shape (1, 3)
    t_a = [
        Tensor(torch.tensor([[9.0]])),
        Tensor(torch.tensor([[25.0]])),
        Tensor(torch.tensor([[49.0]])),
    ]
    a_squared = Block.from_list(t_a, shape=(1, 3))

    # Create b_squared with shape (1, 1) and element_shape=(1,1) for broadcasting
    t_b = [Tensor(torch.tensor([[16.0]]))]
    b_squared = Block.from_list(t_b, shape=(1, 1))

    # Broadcast b_squared along dimension 0 (innermost/columns)
    b_broadcast = ttl.math.broadcast(b_squared, dims=[0])

    # Add them together (should broadcast b to match a's shape)
    result = a_squared + b_broadcast

    # Check result shape
    assert result.shape == (1, 3)


def test_broadcast_multiple_dims():
    """Test broadcast along multiple dimensions."""
    # Create a (1, 1) block with element_shape=(1,1) for broadcasting both dims
    t1 = [Tensor(torch.tensor([[2.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast along both dimensions
    broadcasted = ttl.math.broadcast(block1, dims=[0, 1])

    # Check that it returns a Block
    assert isinstance(broadcasted, Block)


def test_broadcast_preserves_data():
    """Test that broadcast preserves the original data."""
    # Create a block with specific values and element_shape=(1,1) for broadcasting
    original_value = torch.tensor([[7.0]])
    t1 = [Tensor(original_value.clone())]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast it
    broadcasted = ttl.math.broadcast(block1, dims=[0])

    # Check that the broadcast returns a Block
    assert isinstance(broadcasted, Block)


# Tests for explicit broadcasting requirements


def test_implicit_broadcast_rejected():
    """Test that implicit broadcasting is rejected and requires explicit broadcast()."""
    # Create blocks of different shapes
    # Block A: (1, 2) - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Block B: (1, 1) - single tile with element_shape=(1,1) (would need broadcasting)
    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Implicit broadcasting should be rejected
    with pytest.raises(ValueError, match="Use broadcast\\(\\) to expand operands"):
        result = block_a + block_b

    # Explicit broadcasting should work
    broadcasted_b = ttl.math.broadcast(block_b, dims=[0])
    result = block_a + broadcasted_b
    assert result._shape == (1, 2)


def test_implicit_broadcast_different_shapes():
    """Test that implicit broadcasting is rejected for mismatched shapes."""
    # Block A: (2, 1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Block B: (1, 2)
    t_b = [
        Tensor(torch.tensor([[5.0]])),
        Tensor(torch.tensor([[7.0]])),
    ]
    block_b = Block.from_list(t_b, shape=(1, 2))

    # Implicit broadcasting should be rejected
    with pytest.raises(ValueError, match="Use broadcast\\(\\) to expand operands"):
        result = block_a * block_b

    # Explicit broadcasting of both should work.
    # block_a (2,1): broadcast cols (innermost, dims=[-1]); block_b (1,2): broadcast rows (outermost, dims=[0])
    broadcasted_a = ttl.math.broadcast(block_a, dims=[-1])
    broadcasted_b = ttl.math.broadcast(block_b, dims=[0])

    # Can't combine two broadcasts together (ambiguous which should expand first)
    with pytest.raises(ValueError, match="both operands have pending broadcast"):
        result = broadcasted_a * broadcasted_b


def test_matching_shapes_allowed():
    """Test that operations with matching shapes work without broadcast."""
    # Both blocks have shape (1, 2)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    t_b = [
        Tensor(torch.tensor([[10.0]])),
        Tensor(torch.tensor([[30.0]])),
    ]
    block_b = Block.from_list(t_b, shape=(1, 2))

    # This should work - shapes match exactly
    result = block_a + block_b
    assert isinstance(result, Block)
    assert result.shape == (1, 2)


def test_broadcast_on_wrong_dimension_rejected():
    """Test that broadcasting on a dimension with element size != 1 is rejected."""
    # Block with shape (2, 1) and element_shape=(2, 1) - cannot broadcast on dimension 0
    # (outermost/rows, which has element size 2).
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Try to broadcast on dimension 0 (outermost/rows for 2D), which has element size 2
    with pytest.raises(
        ValueError,
        match="Cannot broadcast along dimension 0: dimension must have element size 1",
    ):
        ttl.math.broadcast(block_a, dims=[0])


def test_broadcast_out_of_range_rejected():
    """Test that broadcasting on non-existent dimension is rejected."""
    # Block with shape (1, 1) - only has dimensions 0 and 1
    t_a = [Tensor(torch.tensor([[1.0]]))]
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

    Tests the three forms:
    1) result = a * broadcast(b, dims=[-1]) - explicit broadcast with dims
    2) result = a * broadcast(b, y_unused, dims=[-1]) - explicit with unused output hint
    3) w = broadcast(b, dims=[-1]); result = a * w - intermediate variable

    Note: Implicit broadcasting is no longer supported - must use explicit broadcast().
    """
    # Setup: 'a' is MxN (2x3) and 'b' is Mx1 (2x1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),  # Row 0, Col 0
        Tensor(torch.tensor([[3.0]])),  # Row 0, Col 1
        Tensor(torch.tensor([[5.0]])),  # Row 0, Col 2
        Tensor(torch.tensor([[7.0]])),  # Row 1, Col 0
        Tensor(torch.tensor([[9.0]])),  # Row 1, Col 1
        Tensor(torch.tensor([[11.0]])),  # Row 1, Col 2
    ]
    block_a = Block.from_list(t_a, shape=(2, 3))

    t_b = [
        Tensor(torch.tensor([[2.0]])),  # Row 0, Col 0
        Tensor(torch.tensor([[3.0]])),  # Row 1, Col 0
    ]
    block_b = Block.from_list(t_b, shape=(2, 1))

    # Form 1: Explicit broadcast with dims (dims=[-1]=innermost/columns)
    result1 = block_a * ttl.math.broadcast(block_b, dims=[-1])

    # Form 2: Explicit broadcast with unused output hint (None since we can't create a DFB here)
    result2 = block_a * ttl.math.broadcast(block_b, None, dims=[-1])

    # Form 3: Store broadcast result first, then use it
    broadcast_b = ttl.math.broadcast(block_b, dims=[-1])
    result3 = block_a * broadcast_b

    # All forms should produce the same shape
    assert result1.shape == (2, 3)
    assert result2.shape == (2, 3)
    assert result3.shape == (2, 3)


def test_broadcast_form1_direct_implicit():
    """Test that implicit broadcasting is rejected (was form 1).

    Note: Direct implicit broadcast is no longer supported. Use explicit broadcast().
    """
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Implicit broadcast should be rejected
    with pytest.raises(ValueError, match="Use broadcast\\(\\) to expand operands"):
        result = block_a * block_b

    # Explicit form still works
    result = block_a * ttl.math.broadcast(block_b, dims=[0])
    assert result.shape == (1, 3)


def test_broadcast_form2_explicit_dims():
    """Test form 2: y.store(a * broadcast(b, dims=[0]))."""
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Explicit broadcast with dims parameter (dims=[0]=innermost/columns)
    result = block_a * ttl.math.broadcast(block_b, dims=[0])

    assert result.shape == (1, 3)


def test_broadcast_form3_with_output_hint():
    """Test form 3: y.store(a * broadcast(b, y, dims=[0]))."""
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    t_y = [Tensor(torch.zeros(1, 1)) for _ in range(3)]
    block_y = Block.from_list(t_y, shape=(1, 3))

    # Explicit broadcast with output block hint (unused but accepted)
    result = block_a * ttl.math.broadcast(block_b, block_y, dims=[0])

    assert result.shape == (1, 3)


def test_broadcast_form4_intermediate_store():
    """Test form 4: w = broadcast(b, dims=[0]); result = a * w."""
    # a is (1, 3), b is (1, 1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Store broadcast result in w first (as an intermediate variable, not .store())
    broadcast_b = ttl.math.broadcast(block_b, dims=[0])

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
    """Test reduce_max over rows (outermost dimension 0 for 2D)."""
    # Create a (2, 1) block - two tiles in row dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 5.0]])),
        Tensor(torch.tensor([[3.0, 2.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[2.0, 2.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension 0 (outermost = rows in standard Python indexing)
    result = ttl.math.reduce_max(block_a, scaler, dims=[0])

    # Result should have shape (1, 1) - rows reduced
    assert result.shape == (1, 1)

    # Element-wise max across tiles: max([[1,5], [3,2]]) = [3, 5]
    # Scale by [[2, 2]]: [3, 5] * [2, 2] = [6, 10]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[6.0, 10.0]])
    assert torch.allclose(result_tensor, expected)


def test_reduce_max_cols():
    """Test reduce_max over columns (innermost dimension -1 for 2D)."""
    # Create a (1, 2) block - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 5.0]])),
        Tensor(torch.tensor([[3.0, 2.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension -1 (innermost = column grid dim in standard Python indexing)
    result = ttl.math.reduce_max(block_a, scaler, dims=[-1])

    # Result should have shape (1, 1) - columns reduced
    assert result.shape == (1, 1)

    # Max over cols: max([1, 5], [3, 2]) along innermost dim = [3, 5]
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

    # Element-wise max across all tiles:
    # max([[1,2], [3,4], [5,6], [7,8]]) = [7, 8]
    # Scale by [[0.5, 0.5]]: [7, 8] * [0.5, 0.5] = [3.5, 4.0]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[3.5, 4.0]])
    assert torch.allclose(result_tensor, expected)


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
    """Test reduce_sum over rows (outermost dimension 0 for 2D)."""
    # Create a (2, 1) block - two tiles in row dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[2.0, 2.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension 0 (outermost = rows in standard Python indexing)
    result = ttl.math.reduce_sum(block_a, scaler, dims=[0])

    # Result should have shape (1, 1) - rows reduced
    assert result.shape == (1, 1)

    # Element-wise sum across tiles: sum([[1,2], [3,4]]) = [4, 6]
    # Scale by [[2, 2]]: [4, 6] * [2, 2] = [8, 12]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[8.0, 12.0]])
    assert torch.allclose(result_tensor, expected)


def test_reduce_sum_cols():
    """Test reduce_sum over columns (innermost dimension -1 for 2D)."""
    # Create a (1, 2) block - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Create scaler block (1, 1)
    t_s = [Tensor(torch.tensor([[1.0, 1.0]]))]
    scaler = Block.from_list(t_s, shape=(1, 1))

    # Reduce over dimension -1 (innermost = column grid dim in standard Python indexing)
    result = ttl.math.reduce_sum(block_a, scaler, dims=[-1])

    # Result should have shape (1, 1) - columns reduced
    assert result.shape == (1, 1)

    # Sum over cols: sum([1, 2], [3, 4]) along innermost dim = [4, 6]
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

    # Element-wise sum across all tiles:
    # sum([[1,1], [2,2], [3,3], [4,4]]) = [10, 10]
    # Scale by [[0.1, 0.1]]: [10, 10] * [0.1, 0.1] = [1.0, 1.0]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[1.0, 1.0]])
    assert torch.allclose(result_tensor, expected)


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


# ---------------------------------------------------------------------------
# ND (1-D grid and batched 3-D grid) reduce tests
# ---------------------------------------------------------------------------


def _tile1d(value: float, size: int = 32) -> Tensor:
    """Create a 1-D tile filled with a constant value."""
    return Tensor(torch.full((size,), value))


def _scaler1d(value: float = 1.0) -> "Block":
    """Return a (1,) scaler block with all elements set to value."""
    return Block.from_list([_tile1d(value, 32)], shape=(1,))


def test_reduce_sum_1d_single_tile():
    """reduce_sum on a 1-D (1,) block with a single tile."""
    # One 1-D tile of shape (32,) filled with 2.0
    # Reducing the only grid dimension with 1 tile means no reduction occurs
    block = Block.from_list([_tile1d(2.0)], shape=(1,))
    result = ttl.math.reduce_sum(block, _scaler1d(1.0), dims=[0])
    assert result.shape == (1,)
    out = result.to_list()[0].to_torch()
    # No grid-level reduction (only 1 tile), result is the tile scaled: 2.0 * 1.0
    assert torch.allclose(out, torch.full((32,), 2.0))


def test_reduce_sum_1d_multi_tile():
    """reduce_sum on a 1-D (4,) block reduces all 4 tiles to one."""
    # Four 1-D tiles filled with 3.0
    # Element-wise sum across tiles: 3 + 3 + 3 + 3 = 12 per element
    tiles = [_tile1d(3.0) for _ in range(4)]
    block = Block.from_list(tiles, shape=(4,))
    result = ttl.math.reduce_sum(block, _scaler1d(1.0), dims=[0])
    assert result.shape == (1,)
    out = result.to_list()[0].to_torch()
    # Element-wise sum: 4 tiles * 3.0 = 12.0 per element
    assert torch.allclose(out, torch.full((32,), 12.0))


def test_reduce_sum_1d_scaler():
    """reduce_sum on a 1-D block applies the scaler correctly."""
    block = Block.from_list([_tile1d(1.0)], shape=(1,))
    result = ttl.math.reduce_sum(block, _scaler1d(3.0), dims=[0])
    out = result.to_list()[0].to_torch()
    # Single tile, scaler multiplies each element: 1.0 * 3.0 = 3.0
    assert torch.allclose(out, torch.full((32,), 3.0))


def test_reduce_max_1d_multi_tile():
    """reduce_max on a 1-D (3,) block takes the element-wise max across tiles."""
    # Tiles: all-1, all-5, all-2 -> element-wise max = 5.0 per element
    tiles = [_tile1d(v) for v in [1.0, 5.0, 2.0]]
    block = Block.from_list(tiles, shape=(3,))
    result = ttl.math.reduce_max(block, _scaler1d(1.0), dims=[0])
    assert result.shape == (1,)
    out = result.to_list()[0].to_torch()
    # Element-wise max across 3 tiles: max(1, 5, 2) = 5.0 per element
    assert torch.allclose(out, torch.full((32,), 5.0))


def test_reduce_sum_batched_3d_batch_dim():
    """reduce_sum on a (2, 1, 1) block reducing only the batch dim (outermost = dim 0)."""
    # Two (1,1) batch entries; tile values 4.0 and 6.0 -> grid sum 10.0 per position
    t1 = Tensor(torch.full((1, 1), 4.0))
    t2 = Tensor(torch.full((1, 1), 6.0))
    block = Block.from_list([t1, t2], shape=(2, 1, 1))
    scaler = Block.from_list([Tensor(torch.full((1, 1), 1.0))], shape=(1, 1))
    # dims=[0] = outermost dim (batch) in standard Python indexing
    result = ttl.math.reduce_sum(block, scaler, dims=[0])
    # Batch dim collapsed; spatial dims unchanged: result shape (1, 1, 1)
    assert result.shape == (1, 1, 1)
    out = result.to_list()[0].to_torch()
    # No within-tile reduction (batch dim has no tile axis); grid sum = 4+6 = 10
    assert out[0, 0].item() == pytest.approx(10.0)


def test_reduce_sum_batched_3d_spatial_dim():
    """reduce_sum on a (2, 1, 2) block reducing spatial col dim (innermost = dim -1 for 3D)."""
    # Two batch entries, one row of two column tiles each
    # All tiles filled with 1.0; reducing innermost dim (N=2 -> 1) with within-tile col reduction
    tiles = [Tensor(torch.full((2, 2), 1.0)) for _ in range(4)]  # 2 batch * 1 * 2 tiles
    block = Block.from_list(tiles, shape=(2, 1, 2))
    scaler = Block.from_list([Tensor(torch.full((2, 2), 1.0))], shape=(1, 1))
    # dims=[-1] = innermost dim (spatial col) in standard Python indexing
    result = ttl.math.reduce_sum(block, scaler, dims=[-1])
    # Spatial col dim reduced: result shape (2, 1, 1)
    assert result.shape == (2, 1, 1)


def test_reduce_sum_batched_invalid_dim():
    """reduce_sum on a (2, 1, 1) block rejects dim >= ndim."""
    t = Tensor(torch.full((1, 1), 1.0))
    block = Block.from_list([t, t], shape=(2, 1, 1))
    scaler = Block.from_list([t], shape=(1, 1))
    with pytest.raises(ValueError, match="Cannot reduce along dimension 3"):
        ttl.math.reduce_sum(block, scaler, dims=[3])


def test_transpose_1d_raises():
    """transpose on a 1-D block raises ValueError."""
    block = Block.from_list([_tile1d(1.0)], shape=(1,))
    with pytest.raises(ValueError, match="2-D block grid"):
        ttl.math.transpose(block)


def test_transpose_3d_raises():
    """transpose on a 3-D block raises ValueError."""
    t = Tensor(torch.ones(1, 1))
    block = Block.from_list([t, t], shape=(2, 1, 1))
    with pytest.raises(ValueError, match="2-D block grid"):
        ttl.math.transpose(block)


# ---------------------------------------------------------------------------
# matmul tests
# ---------------------------------------------------------------------------


def _tile(value: float, rows: int = 32, cols: int = 32) -> Tensor:
    """Create a tile filled with a constant value."""
    return Tensor(torch.full((rows, cols), value))


def test_matmul_result_shape_1x1():
    """matmul of (1,1) @ (1,1) produces a (1,1) block."""
    a = Block.from_list([_tile(2.0)], shape=(1, 1))
    b = Block.from_list([_tile(3.0)], shape=(1, 1))
    result = ttl.math.matmul(a, b)
    assert result.shape == (1, 1)
    assert len(result.to_list()) == 1


def test_matmul_result_shape_2x3_times_3x4():
    """matmul of (2,3) @ (3,4) produces a (2,4) block."""
    a = Block.from_list([_tile(1.0)] * 6, shape=(2, 3))
    b = Block.from_list([_tile(1.0)] * 12, shape=(3, 4))
    result = ttl.math.matmul(a, b)
    assert result.shape == (2, 4)
    assert len(result.to_list()) == 8


def test_matmul_values_identity():
    """matmul against an identity-like tile produces the original tile values."""
    rows, cols = 32, 32
    # a tile filled with 5, b tile is identity matrix
    a_tile = Tensor(torch.full((rows, cols), 5.0))
    b_tile = Tensor(torch.eye(cols))

    a = Block.from_list([a_tile], shape=(1, 1))
    b = Block.from_list([b_tile], shape=(1, 1))
    result = ttl.math.matmul(a, b)

    expected = torch.matmul(torch.full((rows, cols), 5.0), torch.eye(cols))
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_matmul_values_accumulation():
    """Each result tile is the sum over k of torch.matmul(a[i,k], b[k,j])."""
    rows, cols = 32, 32
    # (1,2) @ (2,1) -> (1,1); result tile = a[0,0]@b[0,0] + a[0,1]@b[1,0]
    a = Block.from_list([_tile(1.0, rows, cols), _tile(2.0, rows, cols)], shape=(1, 2))
    b = Block.from_list([_tile(3.0, rows, cols), _tile(4.0, rows, cols)], shape=(2, 1))

    result = ttl.math.matmul(a, b)
    assert result.shape == (1, 1)

    # torch.matmul(full(1), full(3)) = cols * 1*3 per element = 12 per element
    # torch.matmul(full(2), full(4)) = cols * 2*4 per element = 32 per element
    # sum = 44 per element
    expected_val = cols * 1.0 * 3.0 + cols * 2.0 * 4.0
    result_tensor = result.to_list()[0].to_torch()
    assert torch.allclose(result_tensor, torch.full((rows, cols), expected_val))


def test_matmul_inner_dim_mismatch_raises():
    """matmul raises RuntimeError when inner tile dimensions do not match."""
    # (1, 3) @ (4, 2): element shapes (32, 96) and (128, 64) are incompatible.
    a = Block.from_list([_tile(1.0)] * 3, shape=(1, 3))
    b = Block.from_list([_tile(1.0)] * 8, shape=(4, 2))
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        ttl.math.matmul(a, b)


def test_matmul_mismatched_inner_dims_raises():
    """matmul raises RuntimeError for a (1,1) @ (2,1) shape mismatch."""
    # Element shapes: (32, 32) and (64, 32) are incompatible for matmul.
    a = Block.from_list([_tile(1.0)], shape=(1, 1))
    b = Block.from_list([_tile(1.0)] * 2, shape=(2, 1))
    with pytest.raises(RuntimeError, match="cannot be multiplied"):
        ttl.math.matmul(a, b)


# ---------------------------------------------------------------------------
# ND-specific tests added to cover gaps identified in the ND support audit
# ---------------------------------------------------------------------------


def test_broadcast_3d_grid_batch_dim():
    """broadcast on a 3D grid block along the batch dimension (outermost = dim 0).

    The batch grid dim has no within-tile axis; the tile content must be
    left unchanged (the existing single tile is simply replicated at the
    grid level by Block.from_list with the same shape).
    """
    # 3D grid (1, 2, 2): 1 batch slice, 2 tile-rows, 2 tile-cols.
    # Each tile is a (32, 32) matrix filled with a distinct value.
    tiles = [Tensor(torch.full((32, 32), float(i))) for i in range(4)]
    block = Block.from_list(tiles, shape=(1, 2, 2))
    # Broadcast along batch dim (outermost = dim 0 in standard Python indexing);
    # the grid dim already has size 1 so this is a no-op at the grid level.
    result = ttl.math.broadcast(block, dims=[0])
    assert result.shape == (1, 2, 2)
    result_tiles = result.to_list()
    for orig, res in zip(tiles, result_tiles):
        assert torch.allclose(
            orig.to_torch(), res.to_torch()
        ), "batch-dim broadcast must not alter tile content"


def test_broadcast_3d_grid_spatial_dim():
    """broadcast on a 3D grid block along a spatial dimension.

    With a (2, 1, 1) block grid and tiles with 1 row, dims=[1] refers to the
    middle (spatial-row) dimension in standard Python indexing. The single
    row of each tile should be replicated to all rows in the target.

    For this 3D block with element_shape=(2, 1, 32), element_shape[1]=1
    so dims=[1] is valid (middle/spatial-row dim has element size 1).
    """
    # Create 1x32 tiles (1 row, 32 cols) that can broadcast along dim 1 (row dimension)
    tile_data = torch.full((1, 32), 7.0)
    tiles = [Tensor(tile_data.clone()), Tensor(tile_data.clone())]
    block = Block.from_list(tiles, shape=(2, 1, 1))
    # Note: This will have element_shape=(2, 1, 32), can broadcast along dim 1
    broadcasted = ttl.math.broadcast(block, dims=[1])

    # After broadcast, the block still has the same value in all elements
    for res_tile in broadcasted.to_list():
        torch_tile = res_tile.to_torch()
        # Tile is 1x32, all values should be 7.0
        assert torch.all(torch_tile == 7.0), "broadcast should preserve tile values"


def test_max_shape_mismatch_raises():
    """math.max raises ValueError when the two blocks have different shapes."""
    a = Block.from_list([_tile(1.0)], shape=(1, 1))
    b = Block.from_list([_tile(1.0)] * 2, shape=(1, 2))
    with pytest.raises(ValueError, match="Shape mismatch"):
        ttl.math.max(a, b)


def test_min_shape_mismatch_raises():
    """math.min raises ValueError when the two blocks have different shapes."""
    a = Block.from_list([_tile(1.0)] * 2, shape=(2, 1))
    b = Block.from_list([_tile(1.0)], shape=(1, 1))
    with pytest.raises(ValueError, match="Shape mismatch"):
        ttl.math.min(a, b)


def test_matmul_batched_3d():
    """Block matmul works correctly for a batched (3D grid) case.

    Two batch slices, each a (M=1, K=1) x (K=1, N=1) tile matmul.
    """
    # Batch size 2; a has shape (2, 1, 1), b has shape (2, 1, 1).
    a_tiles = [_tile(2.0), _tile(3.0)]  # batch 0 and 1
    b_tiles = [_tile(4.0), _tile(5.0)]
    a = Block.from_list(a_tiles, shape=(2, 1, 1))
    b = Block.from_list(b_tiles, shape=(2, 1, 1))
    result = ttl.math.matmul(a, b)
    assert result.shape == (2, 1, 1)
    res_tiles = result.to_list()
    # Each tile is full(v_a) @ full(v_b) = v_a * v_b * 32 per element
    assert torch.allclose(res_tiles[0].to_torch(), torch.full((32, 32), 2.0 * 4.0 * 32))
    assert torch.allclose(res_tiles[1].to_torch(), torch.full((32, 32), 3.0 * 5.0 * 32))


def test_transpose_4d_raises():
    """transpose raises ValueError for a 4-D block grid."""
    tiles = [_tile(1.0)] * 16
    block = Block.from_list(tiles, shape=(2, 2, 2, 2))
    with pytest.raises(ValueError, match="2-D"):
        ttl.math.transpose(block)


def test_transpose_5d_raises():
    """transpose raises ValueError for a 5-D block grid."""
    tiles = [_tile(1.0)] * 16
    block = Block.from_list(tiles, shape=(2, 2, 2, 2, 1))
    with pytest.raises(ValueError, match="2-D"):
        ttl.math.transpose(block)


def test_from_list_to_list_roundtrip_4d():
    """from_list / to_list round-trip for a 4-D block grid (nb=2 batch dims).

    Grid shape (2, 3, 2, 2): 2 batch-0 slices * 3 batch-1 slices *
    2 tile-rows * 2 tile-cols = 24 tiles total.  Each tile is filled with a
    unique value so that any permutation or indexing error in from_list or
    to_list would produce a detectable mismatch.
    """
    shape = (2, 3, 2, 2)
    num_tiles = 2 * 3 * 2 * 2  # 24
    tiles_in = [Tensor(torch.full((32, 32), float(i))) for i in range(num_tiles)]

    block = Block.from_list(tiles_in, shape=shape)
    assert block.shape == shape

    tiles_out = block.to_list()
    assert len(tiles_out) == num_tiles

    for i, (t_in, t_out) in enumerate(zip(tiles_in, tiles_out)):
        assert torch.allclose(
            t_in.to_torch(), t_out.to_torch()
        ), f"Tile {i} mismatch after from_list / to_list round-trip"


def test_1d_broadcast_warning(capsys, compute_thread_context):
    """Test that broadcasting a 1D block generates a hardware warning.

    1D broadcasts are not supported by current hardware, so the simulator
    emits a warning when ttl.math.broadcast() is called on a 1D block.
    This test verifies the warning message is properly displayed.
    """
    # Create a 1D block with shape (1,) - single tile in 1D
    tiles_1d = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_1d = Block.from_list(tiles_1d, shape=(1,))

    assert len(block_1d.shape) == 1, "Test setup: block should be 1D"
    assert block_1d.shape == (1,), "Test setup: block should have shape (1,)"

    # Broadcast the 1D block - this should generate a warning
    result = ttl.math.broadcast(block_1d, dims=[0])

    # Capture stdout output
    captured = capsys.readouterr()

    # Verify the warning message appears
    assert (
        "warning: 1D broadcast is not supported on current hardware" in captured.out
    ), f"Expected 1D broadcast warning not found in output:\n{captured.out}"

    # Verify the broadcast operation still returns a valid Block
    assert isinstance(result, Block)
    assert result.shape == (1,)


def test_threshold_replaces_greater_than():
    """Test that ttl.math.threshold replaces values GREATER THAN threshold.

    Per spec: "For all values greater than specified threshold replace with specified value"
    This is different from torch.threshold which replaces values <= threshold.
    """
    # Create a block with values [1.0, 5.0, 10.0, 15.0]
    tiles = [
        Tensor(torch.tensor([[1.0, 5.0], [10.0, 15.0]])),
    ]
    block = Block.from_list(tiles, shape=(1, 1))

    # Apply threshold: replace values > 8 with 99
    result = ttl.math.threshold(block, threshold=8, value=99)

    # Expected: [1.0, 5.0, 99.0, 99.0]
    # Values 1.0 and 5.0 are <= 8, so they stay unchanged
    # Values 10.0 and 15.0 are > 8, so they become 99.0
    expected = torch.tensor([[1.0, 5.0], [99.0, 99.0]])
    result_tensor = result.to_list()[0].to_torch()

    assert torch.allclose(result_tensor, expected), (
        f"threshold(threshold=8, value=99) failed.\n"
        f"Expected: {expected}\n"
        f"Got: {result_tensor}"
    )
