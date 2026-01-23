# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TensorAccessor implementation for PyTorch tensor access with tile-based indexing.
"""

from typing import Tuple, Union

import torch

from .constants import TILE_SHAPE
from .typedefs import IndexType


# TODO: We only support 2D tensors and 32 x 32 tiles for now, just enough to
# support colman_fused_eltwise_bcast.py and colman_fused_muladd(2).py examples
# from here:
# https://github.com/tenstorrent/tt-mlir/commit/26df1ac1228c7a620981c81e6f1ded6a8cca6cdf#diff-ed227ec06efacdd4fad2d2a4f282428074d6e9b609ad1a2b948d804b1de85441R28
class TensorAccessor:
    """
    A TensorAccessor provides tile-indexed access to 2D PyTorch tensors.

    TensorAccessors abstract tensor access and enforce tile-aligned reads. All indexing
    is done in tile coordinates, where each tile has shape TILE_SHAPE
    (currently (32, 32)). Only 2D tensors are supported.

    Degenerate dimensions (size 1) are allowed - for example, column vectors with
    shape (N, 1) or row vectors with shape (1, N). Non-degenerate dimensions must
    be multiples of the corresponding TILE_SHAPE dimension.

    Usage:
        tensor = torch.randn(128, 128)  # 4x4 tiles
        accessor = TensorAccessor(tensor, index_type=IndexType.TILE)

        # Access using row and column slices in tile coordinates
        tile_data = accessor[slice(0, 1), slice(0, 1)]  # Single tile
        row_data = accessor[slice(0, 1), slice(1, 4)]   # Row of tiles

        # Column vectors with degenerate dimension work too
        col_vec = torch.randn(128, 1)  # 4x1 tiles (column vector)
        accessor = TensorAccessor(col_vec, index_type=IndexType.TILE)
    """

    def __init__(self, tensor: torch.Tensor, index_type: IndexType = IndexType.TILE):
        """Initialize a TensorAccessor with a PyTorch tensor.

        Args:
            tensor: The underlying PyTorch tensor to access. Must be 2D, though
                   dimensions of size 1 are allowed (degenerate dimensions).
            index_type: Must be IndexType.TILE (only supported mode)

        Raises:
            ValueError: If tensor is not 2D
            ValueError: If non-degenerate dimensions are not multiples of the
                       corresponding tile dimensions (degenerate dimensions of size 1 are always valid)
            ValueError: If index_type is not IndexType.TILE
        """
        if index_type != IndexType.TILE:
            raise ValueError(f"Only IndexType.TILE is supported, got {index_type}")

        if len(tensor.shape) != 2:
            raise ValueError(
                f"TensorAccessor only supports 2D tensors, "
                f"got {len(tensor.shape)}D tensor with shape {tensor.shape}"
            )

        self.tensor = tensor
        self.index_type = index_type
        self.shape = tensor.shape

        # Validate non-degenerate dimensions are properly tile-aligned
        # Degenerate dimensions (size 1) are always valid
        for i, dim_size in enumerate(self.shape):
            if dim_size == 1:
                continue
            if dim_size % TILE_SHAPE[i] != 0:
                raise ValueError(
                    f"Tensor dimension {i} has size {dim_size} which is not "
                    f"a multiple of tile dimension {TILE_SHAPE[i]} from TILE_SHAPE={TILE_SHAPE}"
                )

    # The slices are restricted to the use cases seen in the examples here:
    #  https://github.com/tenstorrent/tt-mlir/commit/26df1ac1228c7a620981c81e6f1ded6a8cca6cdf#diff-ed227ec06efacdd4fad2d2a4f282428074d6e9b609ad1a2b948d804b1de85441R28
    def _validate_slice_format(self, s: slice, dimension_name: str) -> None:
        """Validate that a slice has the required format for tile indexing.
        This currently supports only slices with explicit start and stop values,
        and no step value.

        Args:
            s: The slice to validate
            dimension_name: Name of the dimension (for error messages)

        Raises:
            ValueError: If slice format is not supported
        """
        if s.start is None:
            raise ValueError(
                f"Slice {dimension_name} must have explicit start value, got slice({s.start}, {s.stop}, {s.step})"
            )

        if s.stop is None:
            raise ValueError(
                f"Slice {dimension_name} must have explicit stop value, got slice({s.start}, {s.stop}, {s.step})"
            )

        if s.step is not None:
            raise ValueError(
                f"Slice {dimension_name} must not have step value, got slice({s.start}, {s.stop}, {s.step}). Only simple slices are supported."
            )

    def _normalize_index(self, index: Union[int, slice]) -> slice:
        """Convert int index to slice to preserve 2D shape, or return slice as-is.

        Args:
            index: Either an int (converted to slice of length 1) or a slice

        Returns:
            A slice object
        """
        if isinstance(index, int):
            return slice(index, index + 1)
        return index

    def __getitem__(self, key):
        """Access tensor data using tile-based indexing.

        Args:
            key: Tuple of (row_index, col_index) in tile coordinates where each can be:
                 - slice with explicit start and stop (step not supported)
                 - int which will be converted to a slice of length 1 to preserve 2D shape

        Returns:
            Tensor data corresponding to the requested tiles (always 2D)

        Examples:
            accessor[slice(0, 1), slice(0, 1)] -> Single tile at (0, 0)
            accessor[slice(0, 1), slice(1, 4)] -> First row, columns 1-3
            accessor[slice(0, 2), 0] -> First column, rows 0-1 (returns 2D with shape (64, 32))
            accessor[0, slice(0, 2)] -> First row, columns 0-1 (returns 2D with shape (32, 64))
        """
        row_index, col_index = key

        # Convert int indices to slices of length 1 to preserve 2D shape
        row_slice = self._normalize_index(row_index)
        col_slice = self._normalize_index(col_index)

        # Validate slice format
        self._validate_slice_format(row_slice, "row")
        self._validate_slice_format(col_slice, "col")

        # Convert tile slices to element slices
        row_start = row_slice.start * TILE_SHAPE[0]
        row_stop = row_slice.stop * TILE_SHAPE[0]

        col_start = col_slice.start * TILE_SHAPE[1]
        col_stop = col_slice.stop * TILE_SHAPE[1]

        return self.tensor[slice(row_start, row_stop), slice(col_start, col_stop)]

    def __setitem__(self, key, value: torch.Tensor) -> None:
        """Set tensor data using tile-based indexing.

        Args:
            key: Tuple of (row_index, col_index) in tile coordinates where each can be:
                 - slice with explicit start and stop (step not supported)
                 - int which will be converted to a slice of length 1
            value: Tensor data to set at the specified tiles
        """
        row_index, col_index = key

        # Convert int indices to slices of length 1
        row_slice = self._normalize_index(row_index)
        col_slice = self._normalize_index(col_index)

        # Validate slice format
        self._validate_slice_format(row_slice, "row")
        self._validate_slice_format(col_slice, "col")

        # Convert tile slices to element slices
        row_start = row_slice.start * TILE_SHAPE[0]
        row_stop = row_slice.stop * TILE_SHAPE[0]

        col_start = col_slice.start * TILE_SHAPE[1]
        col_stop = col_slice.stop * TILE_SHAPE[1]

        self.tensor[slice(row_start, row_stop), slice(col_start, col_stop)] = value

    def get_tiles(self, row_slice: slice, col_slice: slice) -> torch.Tensor:
        """Return the element tensor corresponding to the provided tile slices.

        This is a small helper to let callers request tiles using two explicit
        slice arguments rather than constructing a tuple `key`. It performs the
        same validation and conversion to element indices used by
        `__getitem__`.
        """
        self._validate_slice_format(row_slice, "row")
        self._validate_slice_format(col_slice, "col")

        row_start = row_slice.start * TILE_SHAPE[0]
        row_stop = row_slice.stop * TILE_SHAPE[0]

        col_start = col_slice.start * TILE_SHAPE[1]
        col_stop = col_slice.stop * TILE_SHAPE[1]

        return self.tensor[slice(row_start, row_stop), slice(col_start, col_stop)]

    def set_tiles(
        self, row_slice: slice, col_slice: slice, value: torch.Tensor
    ) -> None:
        """Set the element tensor corresponding to the provided tile slices."""
        self._validate_slice_format(row_slice, "row")
        self._validate_slice_format(col_slice, "col")

        row_start = row_slice.start * TILE_SHAPE[0]
        row_stop = row_slice.stop * TILE_SHAPE[0]

        col_start = col_slice.start * TILE_SHAPE[1]
        col_stop = col_slice.stop * TILE_SHAPE[1]

        self.tensor[slice(row_start, row_stop), slice(col_start, col_stop)] = value

    def get_tile_shape(self) -> Tuple[int, int]:
        """Get the tensor shape in tiles rather than elements.

        For dimensions of size 1, returns 1 (not divided by TILE_SHAPE).
        """
        tile_rows = 1 if self.shape[0] == 1 else self.shape[0] // TILE_SHAPE[0]
        tile_cols = 1 if self.shape[1] == 1 else self.shape[1] // TILE_SHAPE[1]
        return (tile_rows, tile_cols)

    def validate_tile_coordinates(self, key: Tuple[slice, slice]) -> bool:
        """Validate that the given coordinates are within tile bounds."""
        row_slice, col_slice = key

        # First validate slice format
        try:
            self._validate_slice_format(row_slice, "row")
            self._validate_slice_format(col_slice, "col")
        except ValueError:
            return False

        tile_shape = self.get_tile_shape()

        # Validate row slice bounds
        row_start = row_slice.start
        row_stop = row_slice.stop
        if row_start < 0 or row_stop > tile_shape[0] or row_start >= row_stop:
            return False

        # Validate column slice bounds
        col_start = col_slice.start
        col_stop = col_slice.stop
        if col_start < 0 or col_stop > tile_shape[1] or col_start >= col_stop:
            return False

        return True

    def __repr__(self) -> str:
        return (
            f"TensorAccessor(tensor_shape={self.shape}, "
            f"tile_shape={self.get_tile_shape()})"
        )
