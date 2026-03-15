# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layout creation utilities for tensor distribution across cores."""

from dataclasses import dataclass
from typing import List

from ttl.dialects import ttcore

from .constants import DEFAULT_TILE_SIZE
from .dtype_utils import tensor_dtype_to_ttcore_datatype


@dataclass(frozen=True)
class LayoutConfig:
    """Configuration for TTL layout creation. Supports L1/DRAM interleaved tiled layouts."""

    logical_shape: List[int]
    grid: List[int]
    dtype: str


# BufferType enum values (match TTLOpsEnums.td)
BUFFER_TYPE_L1 = 1

# TensorMemoryLayout enum values (match TTLOpsEnums.td)
TENSOR_MEMORY_LAYOUT_INTERLEAVED = 0


def create_layout(ctx, config: LayoutConfig):
    """
    Create a TTLLayoutAttr for L1 interleaved tiled tensors.

    Supports: L1/DRAM memory, Interleaved layout, tiled (32x32 tiles).

    Args:
        ctx: MLIR context
        config: Configuration with logical_shape, grid, and dtype

    Returns:
        LayoutAttr

    Raises:
        ValueError: If configuration is unsupported
    """
    if len(config.logical_shape) < 2:
        raise ValueError(
            f"Tensors must have at least 2 dimensions, got shape {config.logical_shape}"
        )

    if len(config.grid) != 2:
        raise ValueError(f"Only 2D grids supported, got grid {config.grid}")

    # config.grid is (cols, rows) from tt-lang API, but MLIR expects (rows, cols)
    grid_cols, grid_rows = config.grid
    mlir_grid = [grid_rows, grid_cols]

    ttcore_dtype = tensor_dtype_to_ttcore_datatype(config.dtype)
    element_type = ttcore.ir.TileType.get(
        ctx, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE, ttcore_dtype
    )

    # Import ttl.ir from our _ttlang extension module
    from ttl._mlir_libs._ttlang import ttl_ir

    return ttl_ir.LayoutAttr.get(
        ctx,
        config.logical_shape,
        element_type,
        BUFFER_TYPE_L1,
        mlir_grid,
        TENSOR_MEMORY_LAYOUT_INTERLEAVED,
    )
