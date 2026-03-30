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
    """Configuration for TTL layout creation."""

    logical_shape: List[int]
    grid: List[int]
    dtype: str
    memory_layout: int = 0  # Default: TENSOR_MEMORY_LAYOUT_INTERLEAVED


# BufferType enum values (match TTLOpsEnums.td)
BUFFER_TYPE_L1 = 1

# TensorMemoryLayout enum values (match TTLOpsEnums.td)
TENSOR_MEMORY_LAYOUT_INTERLEAVED = 0
TENSOR_MEMORY_LAYOUT_SINGLE_BANK = 1
TENSOR_MEMORY_LAYOUT_HEIGHT_SHARDED = 2
TENSOR_MEMORY_LAYOUT_WIDTH_SHARDED = 3
TENSOR_MEMORY_LAYOUT_BLOCK_SHARDED = 4

# Map from TTNN memory layout string representations to our enum values.
_TTNN_MEMORY_LAYOUT_MAP = {
    "INTERLEAVED": TENSOR_MEMORY_LAYOUT_INTERLEAVED,
    "SINGLE_BANK": TENSOR_MEMORY_LAYOUT_SINGLE_BANK,
    "HEIGHT_SHARDED": TENSOR_MEMORY_LAYOUT_HEIGHT_SHARDED,
    "WIDTH_SHARDED": TENSOR_MEMORY_LAYOUT_WIDTH_SHARDED,
    "BLOCK_SHARDED": TENSOR_MEMORY_LAYOUT_BLOCK_SHARDED,
}


def detect_memory_layout(tensor) -> int:
    """Detect TensorMemoryLayout enum value from a TTNN tensor."""
    mem_config = tensor.memory_config()
    if hasattr(mem_config, "memory_layout"):
        layout_str = str(mem_config.memory_layout)
        for key, value in _TTNN_MEMORY_LAYOUT_MAP.items():
            if key in layout_str:
                return value
    return TENSOR_MEMORY_LAYOUT_INTERLEAVED


def create_layout(ctx, config: LayoutConfig):
    """
    Create a TTLLayoutAttr for tiled tensors.

    Args:
        ctx: MLIR context
        config: Configuration with logical_shape, grid, dtype, and memory_layout

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
        config.memory_layout,
    )
