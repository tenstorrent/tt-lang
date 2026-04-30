# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for ttl.math and ttl.block module functions.

Includes tests for ttl.block.broadcast and verification of explicit broadcasting requirements.
"""

import pytest
import torch

from sim import ttl
from sim.dfb import Block
from sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor


def test_broadcast_basic():
    """Test basic broadcast operation expanding a (1, 1) block to (3, 1)."""
    t1 = [Tensor(torch.tensor([[5.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast along dimension 0 (outermost/rows) to get (3, 1)
    broadcasted = ttl.block.broadcast(block1, dims=[0], shape=(3, 1))

    assert isinstance(broadcasted, Block)
    assert broadcasted.shape == (3, 1)
    # All 3 tiles should have the same value 5.0
    for tile in broadcasted.to_list():
        assert torch.allclose(tile.to_torch(), torch.tensor([[5.0]]))


def test_broadcast_with_operation():
    """Test broadcast in the context of an operation."""
    # Block A: (1, 2) - two tiles in column dimension
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    # Block B: (1, 1) - single tile
    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Broadcast B along dim -1 (innermost/columns) to match A's shape (1, 2)
    broadcasted_b = ttl.block.broadcast(block_b, dims=[-1], shape=(1, 2))
    result = block_a + broadcasted_b

    assert isinstance(result, Block)
    assert result.shape == (1, 2)
    # Values: [1+10, 3+10] = [11, 13]
    assert torch.allclose(result.to_list()[0].to_torch(), torch.tensor([[11.0]]))
    assert torch.allclose(result.to_list()[1].to_torch(), torch.tensor([[13.0]]))


def test_broadcast_example_from_spec():
    """Test the broadcast example from the specification.

    From spec: y = ttl.math.sqrt(a_squared + ttl.block.broadcast(b_squared, dims=[-1], shape=(1, N)))
    Where a_squared has shape (1, N) and b_squared has shape (1, 1)
    """
    # Create a_squared with shape (1, 3)
    t_a = [
        Tensor(torch.tensor([[9.0]])),
        Tensor(torch.tensor([[25.0]])),
        Tensor(torch.tensor([[49.0]])),
    ]
    a_squared = Block.from_list(t_a, shape=(1, 3))

    # Create b_squared with shape (1, 1)
    t_b = [Tensor(torch.tensor([[16.0]]))]
    b_squared = Block.from_list(t_b, shape=(1, 1))

    # Broadcast b_squared along innermost dim to match a_squared's shape (1, 3)
    b_broadcast = ttl.block.broadcast(b_squared, dims=[-1], shape=(1, 3))

    result = a_squared + b_broadcast

    assert result.shape == (1, 3)
    # Values: [9+16, 25+16, 49+16] = [25, 41, 65]
    assert torch.allclose(result.to_list()[0].to_torch(), torch.tensor([[25.0]]))
    assert torch.allclose(result.to_list()[1].to_torch(), torch.tensor([[41.0]]))
    assert torch.allclose(result.to_list()[2].to_torch(), torch.tensor([[65.0]]))


def test_broadcast_multiple_dims():
    """Test broadcast along multiple dimensions."""
    t1 = [Tensor(torch.tensor([[2.0]]))]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast along both dimensions to (3, 4)
    broadcasted = ttl.block.broadcast(block1, dims=[0, 1], shape=(3, 4))

    assert isinstance(broadcasted, Block)
    assert broadcasted.shape == (3, 4)
    # All 12 tiles should have value 2.0
    for tile in broadcasted.to_list():
        assert torch.allclose(tile.to_torch(), torch.tensor([[2.0]]))


def test_broadcast_preserves_data():
    """Test that broadcast preserves the original data in expanded tiles."""
    original_value = torch.tensor([[7.0]])
    t1 = [Tensor(original_value.clone())]
    block1 = Block.from_list(t1, shape=(1, 1))

    # Broadcast along outermost dim to (2, 1)
    broadcasted = ttl.block.broadcast(block1, dims=[0], shape=(2, 1))

    assert isinstance(broadcasted, Block)
    assert broadcasted.shape == (2, 1)
    for tile in broadcasted.to_list():
        assert torch.allclose(tile.to_torch(), torch.tensor([[7.0]]))


# Tests for explicit broadcasting requirements


def test_implicit_broadcast_rejected():
    """Test that implicit broadcasting is rejected and requires explicit broadcast()."""
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    # Implicit broadcasting should be rejected
    with pytest.raises(ValueError, match="Use broadcast\\(\\) to expand operands"):
        result = block_a + block_b

    # Explicit broadcasting with shape should work
    broadcasted_b = ttl.block.broadcast(block_b, dims=[-1], shape=(1, 2))
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

    # Explicit broadcasting: expand both to (2, 2) first, then multiply
    broadcasted_a = ttl.block.broadcast(block_a, dims=[-1], shape=(2, 2))
    broadcasted_b = ttl.block.broadcast(block_b, dims=[0], shape=(2, 2))
    result = broadcasted_a * broadcasted_b
    assert result.shape == (2, 2)


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
    """Test that broadcasting on a dimension with grid size != 1 is rejected."""
    # Block with shape (2, 1) - cannot broadcast on dimension 0 (grid size is 2, not 1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    with pytest.raises(
        ValueError,
        match="Cannot broadcast along dimension 0: block grid size must be 1",
    ):
        ttl.block.broadcast(block_a, dims=[0], shape=(3, 1))


def test_broadcast_out_of_range_rejected():
    """Test that broadcasting on non-existent dimension is rejected."""
    t_a = [Tensor(torch.tensor([[1.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    with pytest.raises(
        ValueError,
        match="Cannot broadcast along dimension 2.*only 2 dimensions",
    ):
        ttl.block.broadcast(block_a, dims=[2], shape=(1, 1))


# Tests for all different forms of broadcast usage


def test_all_broadcast_forms():
    """Test different broadcast usage patterns.

    1) Inline: result = a * broadcast(b, dims=[-1], shape=(2, 3))
    2) Pre-computed: w = broadcast(b, dims=[-1], shape=(2, 3)); result = a * w
    """
    # Setup: 'a' is (2, 3) and 'b' is (2, 1)
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
        Tensor(torch.tensor([[7.0]])),
        Tensor(torch.tensor([[9.0]])),
        Tensor(torch.tensor([[11.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 3))

    t_b = [
        Tensor(torch.tensor([[2.0]])),
        Tensor(torch.tensor([[3.0]])),
    ]
    block_b = Block.from_list(t_b, shape=(2, 1))

    # Form 1: Inline broadcast with shape
    result1 = block_a * ttl.block.broadcast(block_b, dims=[-1], shape=(2, 3))

    # Form 2: Pre-computed broadcast, then used in operation
    broadcast_b = ttl.block.broadcast(block_b, dims=[-1], shape=(2, 3))
    result2 = block_a * broadcast_b

    assert result1.shape == (2, 3)
    assert result2.shape == (2, 3)


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

    # Explicit form still works with new API
    result = block_a * ttl.block.broadcast(block_b, dims=[-1], shape=(1, 3))
    assert result.shape == (1, 3)


def test_broadcast_form2_explicit_dims():
    """Test form 2: y.store(a * broadcast(b, dims=[-1], shape=(1, 3)))."""
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    result = block_a * ttl.block.broadcast(block_b, dims=[-1], shape=(1, 3))

    assert result.shape == (1, 3)


def test_broadcast_form3_shape_instead_of_hint():
    """Test form 3 (new): y.store(a * broadcast(b, dims=[-1], shape=target_shape))."""
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    result = block_a * ttl.block.broadcast(block_b, dims=[-1], shape=(1, 3))

    assert result.shape == (1, 3)


def test_broadcast_form4_intermediate_store():
    """Test form 4: w = broadcast(b, dims=[-1], shape=(1, 3)); result = a * w."""
    t_a = [
        Tensor(torch.tensor([[1.0]])),
        Tensor(torch.tensor([[3.0]])),
        Tensor(torch.tensor([[5.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 3))

    t_b = [Tensor(torch.tensor([[10.0]]))]
    block_b = Block.from_list(t_b, shape=(1, 1))

    broadcast_b = ttl.block.broadcast(block_b, dims=[-1], shape=(1, 3))
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
    t_a = [
        Tensor(torch.tensor([[1.0, 5.0]])),
        Tensor(torch.tensor([[3.0, 2.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    result = ttl.math.reduce_max(block_a, dims=[0], shape=(1, 1))

    assert result.shape == (1, 1)
    # Element-wise max across tiles: max([[1,5], [3,2]]) = [3, 5]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[3.0, 5.0]])
    assert torch.allclose(result_tensor, expected)


def test_reduce_max_cols():
    """Test reduce_max over columns (innermost dimension -1 for 2D)."""
    t_a = [
        Tensor(torch.tensor([[1.0, 5.0]])),
        Tensor(torch.tensor([[3.0, 2.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    result = ttl.math.reduce_max(block_a, dims=[-1], shape=(1, 1))

    assert result.shape == (1, 1)
    # Max over cols: max([1, 5], [3, 2]) along innermost dim = [3, 5]
    expected = torch.tensor([[3.0, 5.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_max_all():
    """Test reduce_max over all dimensions."""
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
        Tensor(torch.tensor([[5.0, 6.0]])),
        Tensor(torch.tensor([[7.0, 8.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 2))

    result = ttl.math.reduce_max(block_a, dims=[0, 1], shape=(1, 1))

    assert result.shape == (1, 1)
    # Element-wise max across all tiles: max([[1,2], [3,4], [5,6], [7,8]]) = [7, 8]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[7.0, 8.0]])
    assert torch.allclose(result_tensor, expected)


def test_reduce_max_invalid_dims():
    """Test that reduce_max rejects invalid dimensions."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    with pytest.raises(
        ValueError,
        match="Cannot reduce along dimension 2.*only 2 dimensions",
    ):
        ttl.math.reduce_max(block_a, dims=[2], shape=(1, 1))


def test_reduce_max_empty_dims():
    """Test that reduce_max rejects empty dims list."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    with pytest.raises(
        ValueError, match="dims parameter must contain at least one dimension"
    ):
        ttl.math.reduce_max(block_a, dims=[], shape=(1, 1))


# Tests for reduce_sum function


def test_reduce_sum_rows():
    """Test reduce_sum over rows (outermost dimension 0 for 2D)."""
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 1))

    result = ttl.math.reduce_sum(block_a, dims=[0], shape=(1, 1))

    assert result.shape == (1, 1)
    # Element-wise sum across tiles: sum([[1,2], [3,4]]) = [4, 6]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[4.0, 6.0]])
    assert torch.allclose(result_tensor, expected)


def test_reduce_sum_cols():
    """Test reduce_sum over columns (innermost dimension -1 for 2D)."""
    t_a = [
        Tensor(torch.tensor([[1.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(1, 2))

    result = ttl.math.reduce_sum(block_a, dims=[-1], shape=(1, 1))

    assert result.shape == (1, 1)
    # Sum over cols: sum([1, 2], [3, 4]) along innermost dim = [4, 6]
    expected = torch.tensor([[4.0, 6.0]])
    assert torch.allclose(result.to_list()[0].to_torch(), expected)


def test_reduce_sum_all():
    """Test reduce_sum over all dimensions."""
    t_a = [
        Tensor(torch.tensor([[1.0, 1.0]])),
        Tensor(torch.tensor([[2.0, 2.0]])),
        Tensor(torch.tensor([[3.0, 3.0]])),
        Tensor(torch.tensor([[4.0, 4.0]])),
    ]
    block_a = Block.from_list(t_a, shape=(2, 2))

    result = ttl.math.reduce_sum(block_a, dims=[0, 1], shape=(1, 1))

    assert result.shape == (1, 1)
    # Element-wise sum across all tiles: sum([[1,1], [2,2], [3,3], [4,4]]) = [10, 10]
    result_tensor = result.to_list()[0].to_torch()
    expected = torch.tensor([[10.0, 10.0]])
    assert torch.allclose(result_tensor, expected)


def test_reduce_sum_invalid_dims():
    """Test that reduce_sum rejects invalid dimensions."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    with pytest.raises(
        ValueError,
        match="Cannot reduce along dimension 2.*only 2 dimensions",
    ):
        ttl.math.reduce_sum(block_a, dims=[2], shape=(1, 1))


def test_reduce_sum_empty_dims():
    """Test that reduce_sum rejects empty dims list."""
    t_a = [Tensor(torch.tensor([[1.0, 2.0]]))]
    block_a = Block.from_list(t_a, shape=(1, 1))

    with pytest.raises(
        ValueError, match="dims parameter must contain at least one dimension"
    ):
        ttl.math.reduce_sum(block_a, dims=[], shape=(1, 1))


# ---------------------------------------------------------------------------
# ND (1-D grid and batched 3-D grid) reduce tests
# ---------------------------------------------------------------------------


def _tile1d(value: float, size: int = 32) -> Tensor:
    """Create a 1-D tile filled with a constant value."""
    return Tensor(torch.full((size,), value))


def test_reduce_sum_1d_single_tile():
    """reduce_sum on a 1-D (1,) block with a single tile."""
    block = Block.from_list([_tile1d(2.0)], shape=(1,))
    result = ttl.math.reduce_sum(block, dims=[0], shape=(1,))
    assert result.shape == (1,)
    out = result.to_list()[0].to_torch()
    assert torch.allclose(out, torch.full((32,), 2.0))


def test_reduce_sum_1d_multi_tile():
    """reduce_sum on a 1-D (4,) block reduces all 4 tiles to one."""
    tiles = [_tile1d(3.0) for _ in range(4)]
    block = Block.from_list(tiles, shape=(4,))
    result = ttl.math.reduce_sum(block, dims=[0], shape=(1,))
    assert result.shape == (1,)
    out = result.to_list()[0].to_torch()
    # Element-wise sum: 4 tiles * 3.0 = 12.0 per element
    assert torch.allclose(out, torch.full((32,), 12.0))


def test_reduce_max_1d_multi_tile():
    """reduce_max on a 1-D (3,) block takes the element-wise max across tiles."""
    tiles = [_tile1d(v) for v in [1.0, 5.0, 2.0]]
    block = Block.from_list(tiles, shape=(3,))
    result = ttl.math.reduce_max(block, dims=[0], shape=(1,))
    assert result.shape == (1,)
    out = result.to_list()[0].to_torch()
    # Element-wise max across 3 tiles: max(1, 5, 2) = 5.0 per element
    assert torch.allclose(out, torch.full((32,), 5.0))


def test_reduce_sum_batched_3d_batch_dim():
    """reduce_sum on a (2, 1, 1) block reducing only the batch dim (outermost = dim 0)."""
    t1 = Tensor(torch.full((1, 1), 4.0))
    t2 = Tensor(torch.full((1, 1), 6.0))
    block = Block.from_list([t1, t2], shape=(2, 1, 1))
    result = ttl.math.reduce_sum(block, dims=[0], shape=(1, 1, 1))
    assert result.shape == (1, 1, 1)
    out = result.to_list()[0].to_torch()
    assert out[0, 0].item() == pytest.approx(10.0)


def test_reduce_sum_batched_3d_spatial_dim():
    """reduce_sum on a (2, 1, 2) block reducing spatial col dim (innermost = dim -1 for 3D)."""
    tiles = [Tensor(torch.full((2, 2), 1.0)) for _ in range(4)]  # 2 batch * 1 * 2 tiles
    block = Block.from_list(tiles, shape=(2, 1, 2))
    result = ttl.math.reduce_sum(block, dims=[-1], shape=(2, 1, 1))
    assert result.shape == (2, 1, 1)


def test_reduce_sum_batched_invalid_dim():
    """reduce_sum on a (2, 1, 1) block rejects dim >= ndim."""
    t = Tensor(torch.full((1, 1), 1.0))
    block = Block.from_list([t, t], shape=(2, 1, 1))
    with pytest.raises(ValueError, match="Cannot reduce along dimension 3"):
        ttl.math.reduce_sum(block, dims=[3], shape=(1, 1, 1))


def test_transpose_1d_raises():
    """transpose on a 1-D block raises ValueError."""
    block = Block.from_list([_tile1d(1.0)], shape=(1,))
    with pytest.raises(ValueError, match="2-D block grid"):
        ttl.block.transpose(block)


def test_transpose_3d_raises():
    """transpose on a 3-D block raises ValueError."""
    t = Tensor(torch.ones(1, 1))
    block = Block.from_list([t, t], shape=(2, 1, 1))
    with pytest.raises(ValueError, match="2-D block grid"):
        ttl.block.transpose(block)


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
    # Broadcast along batch dim (outermost = dim 0); expand from 1 to 3 batch slices.
    result = ttl.block.broadcast(block, dims=[0], shape=(3, 2, 2))
    assert result.shape == (3, 2, 2)
    # Verify all 12 tiles have the same values as the original 4 tiles (replicated 3 times)
    result_tiles = result.to_list()
    assert len(result_tiles) == 12  # 3 batch * 2 * 2
    for i, res in enumerate(result_tiles):
        orig = tiles[i % 4]
        assert torch.allclose(
            orig.to_torch(), res.to_torch()
        ), "batch-dim broadcast must replicate tile content"


def test_broadcast_3d_grid_spatial_dim():
    """broadcast on a 3D grid block along a spatial dimension (middle dim)."""
    tiles = [Tensor(torch.full((32, 32), 7.0)) for _ in range(2)]
    block = Block.from_list(tiles, shape=(2, 1, 1))
    # Broadcast middle dim from 1 to 3
    broadcasted = ttl.block.broadcast(block, dims=[1], shape=(2, 3, 1))

    assert broadcasted.shape == (2, 3, 1)
    for res_tile in broadcasted.to_list():
        assert torch.all(
            res_tile.to_torch() == 7.0
        ), "broadcast should replicate tile values"


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
        ttl.block.transpose(block)


def test_transpose_5d_raises():
    """transpose raises ValueError for a 5-D block grid."""
    tiles = [_tile(1.0)] * 16
    block = Block.from_list(tiles, shape=(2, 2, 2, 2, 1))
    with pytest.raises(ValueError, match="2-D"):
        ttl.block.transpose(block)


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
    emits a warning when ttl.block.broadcast() is called on a 1D block.
    This test verifies the warning message is properly displayed.
    """
    # Create a 1D block with shape (1,) - single 1D tile
    tiles_1d = [Tensor(torch.tensor([1.0, 2.0]))]
    block_1d = Block.from_list(tiles_1d, shape=(1,))

    assert len(block_1d.shape) == 1, "Test setup: block should be 1D"
    assert block_1d.shape == (1,), "Test setup: block should have shape (1,)"

    # Broadcast the 1D block to shape (3,) - this should generate a warning
    result = ttl.block.broadcast(block_1d, dims=[0], shape=(3,))

    # Capture stdout output
    captured = capsys.readouterr()

    # Verify the warning message appears
    assert (
        "warning: 1D broadcast is not supported on current hardware" in captured.out
    ), f"Expected 1D broadcast warning not found in output:\n{captured.out}"

    # Verify the broadcast operation still returns a valid Block
    assert isinstance(result, Block)
    assert result.shape == (3,)


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


# Tests for Row-Major layout restriction on broadcast and reduce


def _make_row_major_block(shape: tuple) -> Block:
    """Create a block backed by Row-Major layout tensors."""
    import math

    total = math.prod(shape)
    tiles = [Tensor(torch.tensor([[float(i)]]), ROW_MAJOR_LAYOUT) for i in range(total)]
    return Block.from_list(tiles, shape=shape)


def test_broadcast_row_major_rejected():
    """Test that broadcast raises an error for Row-Major layout blocks."""
    block = _make_row_major_block((1, 1))
    with pytest.raises(
        ValueError, match="broadcast is not supported for Row-Major layout"
    ):
        ttl.block.broadcast(block, dims=[0], shape=(3, 1))


def test_reduce_max_row_major_rejected():
    """Test that reduce_max raises an error for Row-Major layout blocks."""
    block = _make_row_major_block((2, 1))
    with pytest.raises(
        ValueError, match="reduce is not supported for Row-Major layout"
    ):
        ttl.math.reduce_max(block, dims=[0], shape=(1, 1))


def test_reduce_sum_row_major_rejected():
    """Test that reduce_sum raises an error for Row-Major layout blocks."""
    block = _make_row_major_block((2, 1))
    with pytest.raises(
        ValueError, match="reduce is not supported for Row-Major layout"
    ):
        ttl.math.reduce_sum(block, dims=[0], shape=(1, 1))


# ---------------------------------------------------------------------------
# fill tests
# ---------------------------------------------------------------------------


def test_fill_basic():
    """fill creates a block of the given shape filled with the specified value."""
    result = ttl.block.fill(3.0, shape=(2, 3))
    assert isinstance(result, Block)
    assert result.shape == (2, 3)
    for tile in result.to_list():
        assert (tile.to_torch() == 3.0).all()


def test_fill_single_tile():
    """fill with shape (1, 1) creates a single-tile block."""
    result = ttl.block.fill(0.0, shape=(1, 1))
    assert result.shape == (1, 1)
    assert (result.to_list()[0].to_torch() == 0.0).all()


def test_fill_requires_2d():
    """fill rejects shapes with fewer than 2 dimensions."""
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        ttl.block.fill(1.0, shape=(4,))


# ---------------------------------------------------------------------------
# squeeze / unsqueeze tests
# ---------------------------------------------------------------------------


def test_squeeze_basic():
    """squeeze removes a size-1 dimension from a block."""
    t = Tensor(torch.tensor([[1.0]]))
    block = Block.from_list([t], shape=(1, 1, 1))
    result = ttl.block.squeeze(block, dims=[0])
    assert result.shape == (1, 1)


def test_squeeze_negative_dim():
    """squeeze supports negative dimension indices."""
    t = Tensor(torch.tensor([[1.0]]))
    block = Block.from_list([t], shape=(1, 1, 1))
    result = ttl.block.squeeze(block, dims=[-1])
    assert result.shape == (1, 1)


def test_squeeze_non_size1_rejected():
    """squeeze rejects a dimension whose grid size is not 1."""
    t_a = [Tensor(torch.tensor([[1.0]])), Tensor(torch.tensor([[2.0]]))]
    block = Block.from_list(t_a, shape=(2, 1))
    with pytest.raises(ValueError, match="grid size is.*expected 1"):
        ttl.block.squeeze(block, dims=[0])


def test_unsqueeze_basic():
    """unsqueeze inserts a size-1 dimension into a block."""
    t_a = [Tensor(torch.tensor([[1.0]])), Tensor(torch.tensor([[2.0]]))]
    block = Block.from_list(t_a, shape=(1, 2))
    result = ttl.block.unsqueeze(block, dims=[0])
    assert result.shape == (1, 1, 2)


def test_unsqueeze_negative_dim():
    """unsqueeze with dim=-1 inserts a size-1 dimension at the end."""
    t_a = [Tensor(torch.tensor([[1.0]])), Tensor(torch.tensor([[2.0]]))]
    block = Block.from_list(t_a, shape=(1, 2))
    result = ttl.block.unsqueeze(block, dims=[-1])
    assert result.shape == (1, 2, 1)


def test_squeeze_unsqueeze_roundtrip():
    """unsqueeze followed by squeeze at the same position is a no-op."""
    t_a = [Tensor(torch.tensor([[float(i)]])) for i in range(6)]
    block = Block.from_list(t_a, shape=(2, 3))
    expanded = ttl.block.unsqueeze(block, dims=[1])
    assert expanded.shape == (2, 1, 3)
    recovered = ttl.block.squeeze(expanded, dims=[1])
    assert recovered.shape == (2, 3)
    # Tile data must be preserved
    for orig, rec in zip(block.to_list(), recovered.to_list()):
        assert torch.allclose(orig.to_torch(), rec.to_torch())


def test_squeeze_multi_dims():
    """squeeze with dims list removes multiple size-1 dimensions at once."""
    t = Tensor(torch.tensor([[1.0]]))
    block = Block.from_list([t], shape=(1, 1, 1, 1))
    result = ttl.block.squeeze(block, dims=[0, 2])
    assert result.shape == (1, 1)


def test_squeeze_multi_dims_negative():
    """squeeze with negative dims removes size-1 dimensions relative to original shape."""
    tiles = [Tensor(torch.tensor([[float(i)]])) for i in range(2)]
    block = Block.from_list(tiles, shape=(1, 2, 1))
    result = ttl.block.squeeze(block, dims=[-1, -3])
    assert result.shape == (2,)
    for orig, rec in zip(block.to_list(), result.to_list()):
        assert torch.allclose(orig.to_torch(), rec.to_torch())


def test_squeeze_both_dim_and_dims_rejected():
    """squeeze rejects the old single-dim keyword argument 'dim'."""
    t = Tensor(torch.tensor([[1.0]]))
    block = Block.from_list([t], shape=(1, 1))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ttl.block.squeeze(block, dim=0)  # type: ignore[call-arg]


def test_unsqueeze_multi_dims():
    """unsqueeze with dims list inserts multiple size-1 dimensions."""
    tiles = [Tensor(torch.tensor([[float(i)]])) for i in range(2)]
    block = Block.from_list(tiles, shape=(1, 2))
    # dims=[0, 2] means insert 1s at positions 0 and 2 of the resulting shape
    result = ttl.block.unsqueeze(block, dims=[0, 2])
    assert result.shape == (1, 1, 1, 2)


def test_unsqueeze_multi_dims_negative():
    """unsqueeze with negative dims inserts size-1 dimensions relative to resulting shape."""
    tiles = [Tensor(torch.tensor([[float(i)]])) for i in range(2)]
    block = Block.from_list(tiles, shape=(1, 2))
    result = ttl.block.unsqueeze(block, dims=[-1, -3])
    assert result.shape == (1, 1, 2, 1)


def test_unsqueeze_both_dim_and_dims_rejected():
    """unsqueeze rejects the old single-dim keyword argument 'dim'."""
    tiles = [Tensor(torch.tensor([[float(i)]])) for i in range(2)]
    block = Block.from_list(tiles, shape=(1, 2))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ttl.block.unsqueeze(block, dim=0)  # type: ignore[call-arg]
