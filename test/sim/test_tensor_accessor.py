# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test suite for the TensorAccessor class with PyTorch tensors.
"""

import pytest
from python.sim import TensorAccessor, IndexType
from python.sim import torch_utils as tu


class TestTensorAccessor:
    """Test the TensorAccessor class functionality."""

    def test_tensor_accessor_creation(self) -> None:
        """Test creating a TensorAccessor with a properly sized tensor."""
        # Create a tensor that's a multiple of tile dimensions
        tensor = tu.randn(64, 64)
        accessor = TensorAccessor(tensor, index_type=IndexType.TILE)

        assert accessor.shape == (64, 64)
        assert accessor.get_tile_shape() == (2, 2)

    def test_tensor_accessor_invalid_size(self) -> None:
        """Test that TensorAccessor rejects tensors with invalid sizes."""
        # Create tensor that's not a multiple of tile dimensions
        tensor = tu.randn(30, 30)

        with pytest.raises(ValueError, match="not a multiple of tile dimension"):
            TensorAccessor(tensor, index_type=IndexType.TILE)

    def test_tensor_accessor_invalid_index_type(self) -> None:
        """Test that TensorAccessor only accepts IndexType.TILE."""
        tensor = tu.randn(64, 64)

        # This should work
        accessor = TensorAccessor(tensor, index_type=IndexType.TILE)
        assert accessor.index_type == IndexType.TILE

    def test_single_tile_access(self) -> None:
        """Test accessing a single tile."""
        # Create test tensor with known values
        tensor = tu.zeros(64, 64)
        # Fill tile (0,0) with ones
        tensor[0:32, 0:32] = 1.0
        # Fill tile (1,1) with twos
        tensor[32:64, 32:64] = 2.0

        stream = TensorAccessor(tensor)

        # Access tile (0,0) using slice notation
        tile_00 = stream[slice(0, 1), slice(0, 1)]
        assert tile_00.shape == (32, 32)
        assert tu.all_true(tile_00 == 1.0)

        # Access tile (1,1) using slice notation
        tile_11 = stream[slice(1, 2), slice(1, 2)]
        assert tile_11.shape == (32, 32)
        assert tu.all_true(tile_11 == 2.0)

    def test_slice_access(self) -> None:
        """Test accessing multiple tiles with slices."""
        # Create 4x4 tile tensor (128x128 elements)
        tensor = tu.zeros(128, 128)

        # Fill first row of tiles with value 1
        tensor[0:32, :] = 1.0
        # Fill second row of tiles with value 2
        tensor[32:64, :] = 2.0

        stream = TensorAccessor(tensor)

        # Access first row of tiles (slice 0:1, slice 0:4)
        first_row = stream[slice(0, 1), slice(0, 4)]
        assert first_row.shape == (32, 128)
        assert tu.all_true(first_row == 1.0)

        # Access first two rows (slice 0:2, slice 0:4)
        first_two_rows = stream[slice(0, 2), slice(0, 4)]
        assert first_two_rows.shape == (64, 128)
        assert tu.all_true(first_two_rows[0:32, :] == 1.0)
        assert tu.all_true(first_two_rows[32:64, :] == 2.0)

    def test_mixed_indexing(self) -> None:
        """Test accessing single tiles and rows of tiles."""
        tensor = tu.zeros(64, 96)  # 2x3 tiles
        tensor[32:64, 32:64] = 5.0  # Fill tile (1,1) with 5s

        stream = TensorAccessor(tensor)

        # Access row 1, column 1 (single tile)
        tile = stream[slice(1, 2), slice(1, 2)]
        assert tile.shape == (32, 32)
        assert tu.all_true(tile == 5.0)

        # Access row 1, all columns
        row = stream[slice(1, 2), slice(0, 3)]
        assert row.shape == (32, 96)
        assert tu.all_true(row[:, 32:64] == 5.0)
        assert tu.all_true(row[:, 0:32] == 0.0)
        assert tu.all_true(row[:, 64:96] == 0.0)

    def test_setitem(self) -> None:
        """Test setting values through tile indexing."""
        tensor = tu.zeros(64, 64)
        stream = TensorAccessor(tensor)

        # Set tile (0,0) to ones
        ones_tile = tu.ones(32, 32)
        stream[slice(0, 1), slice(0, 1)] = ones_tile

        # Verify the change
        assert tu.all_true(stream[slice(0, 1), slice(0, 1)] == 1.0)
        assert tu.all_true(
            stream[slice(0, 1), slice(1, 2)] == 0.0
        )  # Other tiles unchanged

        # Set a slice of tiles
        twos_block = tu.full((32, 64), 2.0)
        stream[slice(1, 2), slice(0, 2)] = twos_block

        # Verify the slice change
        assert tu.all_true(stream[slice(1, 2), slice(0, 2)] == 2.0)
        assert tu.all_true(
            stream[slice(0, 1), slice(0, 2)] != 2.0
        )  # Other rows unchanged

    def test_validate_tile_coordinates(self) -> None:
        """Test coordinate validation."""
        tensor = tu.zeros(64, 96)  # 2x3 tiles
        stream = TensorAccessor(tensor)

        # Valid coordinates
        assert stream.validate_tile_coordinates((slice(0, 1), slice(0, 1)))
        assert stream.validate_tile_coordinates((slice(1, 2), slice(2, 3)))
        assert stream.validate_tile_coordinates((slice(0, 2), slice(0, 3)))

        # Invalid coordinates
        assert not stream.validate_tile_coordinates(
            (slice(2, 3), slice(0, 1))
        )  # Row out of bounds
        assert not stream.validate_tile_coordinates(
            (slice(0, 1), slice(3, 4))
        )  # Col out of bounds
        assert not stream.validate_tile_coordinates(
            (slice(0, 3), slice(0, 2))
        )  # Row slice too big

    def test_repr(self) -> None:
        """Test string representation."""
        tensor = tu.zeros(96, 64)  # 3x2 tiles
        stream = TensorAccessor(tensor)

        repr_str = repr(stream)
        assert "tensor_shape=torch.Size([96, 64])" in repr_str
        assert "tile_shape=(3, 2)" in repr_str

    def test_real_usage_pattern(self) -> None:
        """Test usage pattern from tt-lang code examples."""
        # Create input tensor like in the examples
        a_in = tu.randn(128, 128)  # 4x4 tiles

        # Create accessor like in colman_fused examples
        a_accessor = TensorAccessor(a_in, index_type=IndexType.TILE)

        # Test access patterns similar to the code
        rt = 0  # tile row
        ct_block = 1  # tile column block
        granularity = 2  # number of tiles

        # Access like: a_slice = a_accessor[slice(rt, rt+1), slice(ct_block*granularity, (ct_block+1)*granularity)]
        a_slice = a_accessor[
            slice(rt, rt + 1),
            slice(ct_block * granularity, (ct_block + 1) * granularity),
        ]

        # Should get 1 row of tiles, 2 columns of tiles -> 32x64 elements
        assert a_slice.shape == (32, 64)

        # Test another pattern
        rt2 = 2
        a_slice2 = a_accessor[slice(rt2, rt2 + 1), slice(0, 4)]  # Full row
        assert a_slice2.shape == (32, 128)

    def test_slice_validation(self) -> None:
        """Test that only supported slice formats are accepted."""
        tensor = tu.zeros(64, 64)
        stream = TensorAccessor(tensor)

        # Valid slice should work
        result = stream[slice(0, 1), slice(0, 1)]
        assert result.shape == (32, 32)

        # slice with None start should fail
        with pytest.raises(ValueError, match="must have explicit start value"):
            stream[slice(None, 1), slice(0, 1)]

        # slice with None stop should fail
        with pytest.raises(ValueError, match="must have explicit stop value"):
            stream[slice(0, None), slice(0, 1)]

        # slice with step should fail
        with pytest.raises(ValueError, match="must not have step value"):
            stream[slice(0, 1, 1), slice(0, 1)]

        # Test validation method
        assert stream.validate_tile_coordinates((slice(0, 1), slice(0, 1))) is True
        assert stream.validate_tile_coordinates((slice(None, 1), slice(0, 1))) is False
        assert stream.validate_tile_coordinates((slice(0, 1, 1), slice(0, 1))) is False

    def test_tensor_dimension_validation(self) -> None:
        """Test that only 2D tensors are accepted."""
        # Valid 2D tensor should work
        tensor_2d = tu.randn(64, 64)
        stream = TensorAccessor(tensor_2d)
        assert stream.get_tile_shape() == (2, 2)

        # 1D tensor should fail
        with pytest.raises(
            ValueError, match="TensorAccessor only supports 2D tensors, got 1D tensor"
        ):
            tensor_1d = tu.randn(64)
            TensorAccessor(tensor_1d)

        # 3D tensor should fail
        with pytest.raises(
            ValueError, match="TensorAccessor only supports 2D tensors, got 3D tensor"
        ):
            tensor_3d = tu.randn(2, 64, 64)
            TensorAccessor(tensor_3d)

        # 4D tensor should fail
        with pytest.raises(
            ValueError, match="TensorAccessor only supports 2D tensors, got 4D tensor"
        ):
            tensor_4d = tu.randn(2, 3, 64, 64)
            TensorAccessor(tensor_4d)
