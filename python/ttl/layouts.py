# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layout creation utilities for tensor distribution across cores."""

from dataclasses import dataclass
from typing import List

from ttmlir.dialects import ttcore, ttnn

from .constants import DEFAULT_TILE_SIZE
from .dtype_utils import tensor_dtype_to_ttcore_datatype


@dataclass(frozen=True)
class TTNNLayoutConfig:
    """Configuration for TTNN layout creation. Supports L1/DRAM interleaved tiled layouts."""

    logical_shape: List[int]
    grid: List[int]
    dtype: str


# TTNN BufferType enum values (from TTNNOpsEnums.td)
_TTNN_BUFFER_TYPE_L1 = 1

# TTNN TensorMemoryLayout enum values (from TTNNOpsEnums.td)
_TTNN_TENSOR_MEMORY_LAYOUT_INTERLEAVED = 0


def create_ttnn_layout(ctx, config: TTNNLayoutConfig):
    """
    Create a TTNNLayoutAttr for L1 interleaved tiled tensors.

    Supports: L1/DRAM memory, Interleaved layout, tiled (32x32 tiles).

    Args:
        ctx: MLIR context
        config: Configuration with logical_shape, grid, and dtype

    Returns:
        TTNNLayoutAttr

    Raises:
        ValueError: If configuration is unsupported
    """
    if len(config.logical_shape) != 2:
        raise ValueError(f"Only 2D tensors supported, got shape {config.logical_shape}")

    if len(config.grid) != 2:
        raise ValueError(f"Only 2D grids supported, got grid {config.grid}")

    # config.grid is (cols, rows) from tt-lang API, but MLIR expects (rows, cols)
    grid_cols, grid_rows = config.grid
    mlir_grid = [grid_rows, grid_cols]

    # logical_shape is (rows, cols), mlir_grid is (rows, cols)
    tensor_rows, tensor_cols = config.logical_shape
    if tensor_rows % grid_rows != 0:
        raise ValueError(
            f"Tensor rows ({tensor_rows}) must be divisible by grid rows ({grid_rows})"
        )
    if tensor_cols % grid_cols != 0:
        raise ValueError(
            f"Tensor cols ({tensor_cols}) must be divisible by grid cols ({grid_cols})"
        )

    ttcore_dtype = tensor_dtype_to_ttcore_datatype(config.dtype)
    element_type = ttcore.ir.TileType.get(
        ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
    )

    grid_attr = ttcore.ir.GridAttr.get(ctx, mlir_grid)

    return ttnn.ir.TTNNLayoutAttr.get(
        ctx,
        config.logical_shape,
        element_type,
        _TTNN_BUFFER_TYPE_L1,
        grid_attr,
        _TTNN_TENSOR_MEMORY_LAYOUT_INTERLEAVED,
    )
