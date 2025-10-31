"""
TensorAccessor implementation for PyTorch tensor access with tile-based indexing.
"""

from typing import Tuple
import torch

from .constants import TILE_SIZE
from .index_type import IndexType

# TODO: We only support 2D tensors and 32 x 32 tiles for now, just enough to 
# support colman_fused_eltwise_bcast.py and colman_fused_muladd(2).py examples
# from here:
# https://github.com/tenstorrent/tt-mlir/commit/26df1ac1228c7a620981c81e6f1ded6a8cca6cdf#diff-ed227ec06efacdd4fad2d2a4f282428074d6e9b609ad1a2b948d804b1de85441R28
class TensorAccessor:
    """
    A TensorAccessor provides tile-indexed access to 2D PyTorch tensors.
    
    TensorAccessors abstract tensor access and enforce tile-aligned reads. All indexing
    is done in tile coordinates, where each tile is TILE_SIZE x TILE_SIZE
    (currently 32 x 32). Only 2D tensors are supported currently.
    
    Usage:
        tensor = torch.randn(128, 128)  # 4x4 tiles, must be 2D
        accessor = TensorAccessor(tensor, index_type=IndexType.TILE)
        
        # Access using row and column slices in tile coordinates
        tile_data = accessor[slice(0, 1), slice(0, 1)]  # Single tile
        row_data = accessor[slice(0, 1), slice(1, 4)]   # Row of tiles
    """
    
    def __init__(self, tensor: torch.Tensor, index_type: IndexType = IndexType.TILE):
        """Initialize a TensorAccessor with a PyTorch tensor.
        
        Args:
            tensor: The underlying PyTorch tensor to access (must be 2D)
            index_type: Must be IndexType.TILE (only supported mode)
            
        Raises:
            ValueError: If tensor is not 2D
            ValueError: If tensor dimensions aren't multiples of TILE_SIZE
            ValueError: If index_type is not IndexType.TILE
        """
        if index_type != IndexType.TILE:
            raise ValueError(f"Only IndexType.TILE is supported, got {index_type}")
        
        if len(tensor.shape) != 2:
            raise ValueError(f"TensorAccessor only supports 2D tensors, got {len(tensor.shape)}D tensor with shape {tensor.shape}")
            
        self.tensor = tensor
        self.index_type = index_type
        self.shape = tensor.shape
        
        # Validate tensor is properly tiled (multiples of TILE_SIZE)
        for i, dim_size in enumerate(self.shape):
            if dim_size % TILE_SIZE != 0:
                raise ValueError(
                    f"Tensor dimension {i} has size {dim_size} which is not "
                    f"a multiple of TILE_SIZE={TILE_SIZE}"
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
            raise ValueError(f"Slice {dimension_name} must have explicit start value, got slice({s.start}, {s.stop}, {s.step})")
        
        if s.stop is None:
            raise ValueError(f"Slice {dimension_name} must have explicit stop value, got slice({s.start}, {s.stop}, {s.step})")
        
        if s.step is not None:
            raise ValueError(f"Slice {dimension_name} must not have step value, got slice({s.start}, {s.stop}, {s.step}). Only simple slices are supported.")
    
    def __getitem__(self, key: Tuple[slice, slice]) -> torch.Tensor:
        """Access tensor data using tile-based indexing.
        
        Args:
            key: Tuple of (row_slice, col_slice) in tile coordinates
                 Both slices must have explicit start and stop values.
                 Step values are not supported.
                 
        Returns:
            Tensor data corresponding to the requested tiles
            
        Examples:
            accessor[slice(0, 1), slice(0, 1)] -> Single tile at (0, 0)
            accessor[slice(0, 1), slice(1, 4)] -> First row, columns 1-3
        """
        row_slice, col_slice = key
        
        # Validate slice format
        self._validate_slice_format(row_slice, "row")
        self._validate_slice_format(col_slice, "col")
        
        # Convert tile slices to element slices
        row_start = row_slice.start * TILE_SIZE
        row_stop = row_slice.stop * TILE_SIZE
        
        col_start = col_slice.start * TILE_SIZE
        col_stop = col_slice.stop * TILE_SIZE
        
        return self.tensor[slice(row_start, row_stop), slice(col_start, col_stop)]
    
    def __setitem__(self, key: Tuple[slice, slice], value: torch.Tensor) -> None:
        """Set tensor data using tile-based indexing.
        
        Args:
            key: Tuple of (row_slice, col_slice) in tile coordinates
                 Both slices must have explicit start and stop values.
                 Step values are not supported.
            value: Tensor data to set at the specified tiles
        """
        row_slice, col_slice = key
        
        # Validate slice format
        self._validate_slice_format(row_slice, "row")
        self._validate_slice_format(col_slice, "col")
        
        # Convert tile slices to element slices
        row_start = row_slice.start * TILE_SIZE
        row_stop = row_slice.stop * TILE_SIZE
        
        col_start = col_slice.start * TILE_SIZE
        col_stop = col_slice.stop * TILE_SIZE
        
        self.tensor[slice(row_start, row_stop), slice(col_start, col_stop)] = value
    
    def get_tile_shape(self) -> Tuple[int, int]:
        """Get the tensor shape in tiles rather than elements.
        """
        return (self.shape[0] // TILE_SIZE, self.shape[1] // TILE_SIZE)
    
    def validate_tile_coordinates(self, key: Tuple[slice, slice]) -> bool:
        """Validate that the given coordinates are within tile bounds.
        """
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
    
    @property
    def tile_size(self) -> int:
        """Get the tile size used by this TensorAccessor."""
        return TILE_SIZE
    
    def __repr__(self) -> str:
        return (
            f"TensorAccessor(tensor_shape={self.shape}, "
            f"tile_shape={self.get_tile_shape()}, "
            f"tile_size={TILE_SIZE})"
        )