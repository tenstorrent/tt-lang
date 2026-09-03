# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layout creation utilities for tensor distribution across cores."""

from dataclasses import dataclass
from typing import Any, List, Tuple

from ttl.dialects import ttcore, ttl as ttl_dialect

from .constants import DEFAULT_TILE_SIZE
from .dtype_utils import (
    TTNN_DTYPE_NAMES_TO_TTCORE_DATATYPES,
    TTNN_ROW_MAJOR_DTYPE_NAMES,
    tensor_dtype_to_ttcore_datatype,
)

BUFFER_TYPE_DRAM = ttl_dialect.BufferType.DRAM
BUFFER_TYPE_L1 = ttl_dialect.BufferType.L1
BUFFER_TYPE_SYSTEM_MEMORY = ttl_dialect.BufferType.SystemMemory
BUFFER_TYPE_L1_SMALL = ttl_dialect.BufferType.L1Small

TENSOR_MEMORY_LAYOUT_INTERLEAVED = ttl_dialect.TensorMemoryLayout.Interleaved
TENSOR_MEMORY_LAYOUT_SINGLE_BANK = ttl_dialect.TensorMemoryLayout.SingleBank
TENSOR_MEMORY_LAYOUT_HEIGHT_SHARDED = ttl_dialect.TensorMemoryLayout.HeightSharded
TENSOR_MEMORY_LAYOUT_WIDTH_SHARDED = ttl_dialect.TensorMemoryLayout.WidthSharded
TENSOR_MEMORY_LAYOUT_BLOCK_SHARDED = ttl_dialect.TensorMemoryLayout.BlockSharded
TENSOR_MEMORY_LAYOUT_ND_SHARDED = ttl_dialect.TensorMemoryLayout.NdSharded

TENSOR_LAYOUT_TILE = "tile"
TENSOR_LAYOUT_ROW_MAJOR = "row_major"

_TTNN_DTYPE_NAMES_BY_TENSOR_LAYOUT = {
    TENSOR_LAYOUT_TILE: tuple(TTNN_DTYPE_NAMES_TO_TTCORE_DATATYPES),
    TENSOR_LAYOUT_ROW_MAJOR: TTNN_ROW_MAJOR_DTYPE_NAMES,
}


@dataclass(frozen=True)
class LayoutConfig:
    """Configuration for TTL layout creation."""

    logical_shape: List[int]
    grid: List[int]
    dtype: Any
    memory_layout: int = TENSOR_MEMORY_LAYOUT_INTERLEAVED
    tile: Tuple[int, int] = (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)
    buffer_type: int = BUFFER_TYPE_L1
    tensor_layout: str = TENSOR_LAYOUT_TILE


def detect_memory_layout(tensor) -> int:
    """Map a TTNN tensor's exact memory layout to the TTL enum value."""
    ttnn = _get_ttnn()
    memory_layout = tensor.memory_config().memory_layout
    supported_layouts = {
        ttnn.TensorMemoryLayout.INTERLEAVED: TENSOR_MEMORY_LAYOUT_INTERLEAVED,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED: TENSOR_MEMORY_LAYOUT_HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED: TENSOR_MEMORY_LAYOUT_WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED: TENSOR_MEMORY_LAYOUT_BLOCK_SHARDED,
        ttnn.TensorMemoryLayout.ND_SHARDED: TENSOR_MEMORY_LAYOUT_ND_SHARDED,
    }
    if memory_layout in supported_layouts:
        return supported_layouts[memory_layout]
    raise ValueError(f"Unsupported TTNN tensor memory layout: {memory_layout}")


def _get_ttnn():
    try:
        import ttnn
    except (ModuleNotFoundError, ImportError) as error:
        raise ValueError(
            "TTNN tensor configuration requires the ttnn module"
        ) from error
    return ttnn


def get_supported_ttnn_dtype_layouts(ttnn_module=None):
    """Return the TTNN dtype and tensor-layout combinations accepted by TTL."""
    if ttnn_module is None:
        ttnn_module = _get_ttnn()
    layout_values = {
        TENSOR_LAYOUT_TILE: ttnn_module.TILE_LAYOUT,
        TENSOR_LAYOUT_ROW_MAJOR: ttnn_module.ROW_MAJOR_LAYOUT,
    }
    return tuple(
        (
            getattr(ttnn_module.DataType, dtype_name),
            layout_values[tensor_layout],
        )
        for tensor_layout, dtype_names in _TTNN_DTYPE_NAMES_BY_TENSOR_LAYOUT.items()
        for dtype_name in dtype_names
    )


def detect_buffer_type(tensor) -> int:
    """Map a TTNN tensor's exact buffer type to the TTL enum value."""
    ttnn = _get_ttnn()
    storage_type = tensor.storage_type()
    if storage_type == ttnn.StorageType.HOST:
        return BUFFER_TYPE_SYSTEM_MEMORY
    if storage_type != ttnn.StorageType.DEVICE:
        raise ValueError(f"Unsupported TTNN tensor storage type: {storage_type}")
    buffer_type = tensor.memory_config().buffer_type
    supported_types = {
        ttnn.BufferType.DRAM: BUFFER_TYPE_DRAM,
        ttnn.BufferType.L1: BUFFER_TYPE_L1,
        ttnn.BufferType.L1_SMALL: BUFFER_TYPE_L1_SMALL,
    }
    if buffer_type in supported_types:
        return supported_types[buffer_type]
    if buffer_type == ttnn.BufferType.TRACE:
        raise ValueError("TTNN trace buffers cannot be tensor arguments")
    raise ValueError(f"Unsupported TTNN buffer type: {buffer_type}")


def detect_tensor_layout(tensor) -> str:
    """Map a TTNN tensor's exact tensor layout to the TTL representation."""
    ttnn = _get_ttnn()
    if tensor.layout == ttnn.TILE_LAYOUT:
        tensor_layout = TENSOR_LAYOUT_TILE
    elif tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        tensor_layout = TENSOR_LAYOUT_ROW_MAJOR
    else:
        raise ValueError(f"Unsupported TTNN tensor layout: {tensor.layout}")
    if (tensor.dtype, tensor.layout) not in get_supported_ttnn_dtype_layouts(ttnn):
        raise ValueError(
            f"Unsupported TTNN dtype/layout combination: {tensor.dtype} with "
            f"{tensor.layout}"
        )
    return tensor_layout


def get_tensor_configuration(tensor) -> tuple:
    """Return tensor properties that affect compilation and runtime descriptors."""
    shape = tuple(tensor.shape)
    padded_shape = tuple(getattr(tensor, "padded_shape", tensor.shape))
    dtype = str(tensor.dtype)
    storage_type = (
        str(tensor.storage_type()) if hasattr(tensor, "storage_type") else "unknown"
    )
    memory_config = tensor.memory_config()
    if hasattr(memory_config, "to_json"):
        serialized_memory_config = memory_config.to_json()
    else:
        serialized_memory_config = (
            str(getattr(memory_config, "buffer_type", "unknown")),
            str(getattr(memory_config, "memory_layout", "unknown")),
            str(getattr(memory_config, "shard_spec", None)),
            str(getattr(memory_config, "nd_shard_spec", None)),
        )
    tensor_layout = str(tensor.layout) if hasattr(tensor, "layout") else "unknown"
    tile = (
        tuple(tensor.get_tile().tile_shape)
        if "TILE" in tensor_layout and hasattr(tensor, "get_tile")
        else None
    )
    return (
        shape,
        padded_shape,
        dtype,
        storage_type,
        serialized_memory_config,
        tensor_layout,
        tile,
    )


def _create_scalar_element_type(ctx, dtype):
    from ttl.ir import BF16Type, F32Type, IntegerType

    if dtype == ttcore.DataType.Float32:
        return F32Type.get(ctx)
    if dtype == ttcore.DataType.BFloat16:
        return BF16Type.get(ctx)
    if dtype == ttcore.DataType.UInt32:
        return IntegerType.get_unsigned(32, ctx)
    if dtype == ttcore.DataType.UInt16:
        return IntegerType.get_unsigned(16, ctx)
    if dtype == ttcore.DataType.UInt8:
        return IntegerType.get_unsigned(8, ctx)
    if dtype == ttcore.DataType.Int32:
        return IntegerType.get_signed(32, ctx)
    raise ValueError(f"Data type {dtype} requires TILE layout")


def create_layout_element_type(ctx, config: LayoutConfig):
    """Create the tensor element type encoded by a layout configuration."""
    ttcore_dtype = tensor_dtype_to_ttcore_datatype(config.dtype)
    if config.tensor_layout == TENSOR_LAYOUT_TILE:
        return ttcore.ir.TileType.get(ctx, config.tile[0], config.tile[1], ttcore_dtype)
    if config.tensor_layout == TENSOR_LAYOUT_ROW_MAJOR:
        return _create_scalar_element_type(ctx, ttcore_dtype)
    raise ValueError(f"Unsupported tensor layout: {config.tensor_layout}")


def create_layout(ctx, config: LayoutConfig):
    """
    Create a TTLLayoutAttr for tiled or row-major tensors.

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

    element_type = create_layout_element_type(ctx, config)

    # Import ttl.ir from our _ttlang extension module
    from ttl._mlir_libs._ttlang import ttl_ir

    return ttl_ir.LayoutAttr.get(
        ctx,
        config.logical_shape,
        element_type,
        config.buffer_type,
        mlir_grid,
        config.memory_layout,
    )
