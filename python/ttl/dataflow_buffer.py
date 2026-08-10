# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dataflow buffer (DFB) operations for inter-thread communication."""

from dataclasses import dataclass
import math
from typing import Any, Optional, Tuple

from ttl.ir import *

from ._src.ttl_ast import syntax
from .constants import DEFAULT_TILE_SIZE, SUPPORTED_TENSOR_BACKED_DFB_MEMORY_LAYOUTS
from .dfb_allocation_group import (
    DFBAllocationGroup,
    _BoundDFBAllocationGroup,
    _bind_current_dfb_allocation_group,
)
from .dtype_utils import normalize_tile_dimensions
from ttl.dialects import ttl

_DFB_DESCRIPTOR_UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class _TensorBackedDFBTensorProperties:
    tile_shape: Tuple[int, int]
    page_size: int
    logical_shard_size_bytes: int


def _validate_tensor_backed_dfb_tensor(
    tensor: Any, *, context: str
) -> _TensorBackedDFBTensorProperties:
    """Validate public TTNN properties used by tensor-backed DFB storage."""
    try:
        memory_config = tensor.memory_config()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"{context} must expose a TTNN memory configuration"
        ) from error

    if "L1" not in str(memory_config.buffer_type):
        raise ValueError(f"{context} must use L1 storage")
    memory_layout = str(memory_config.memory_layout).rsplit(".", maxsplit=1)[-1]
    if memory_layout not in SUPPORTED_TENSOR_BACKED_DFB_MEMORY_LAYOUTS:
        raise ValueError(
            f"{context} must be height-, width-, or block-sharded, "
            f"got {memory_config.memory_layout}"
        )
    if "TILE" not in str(getattr(tensor, "layout", None)):
        raise ValueError(f"{context} must use TILE layout")

    dtype_name = str(tensor.dtype).rsplit(".", maxsplit=1)[-1].lower()
    if dtype_name not in {"bfloat16", "float32"}:
        raise ValueError(
            f"{context} supports BF16 and FP32 tensors, got {tensor.dtype}"
        )

    try:
        tile = tensor.get_tile()
        tile_shape = tuple(int(dimension) for dimension in tile.tile_shape)
        page_size = int(tile.get_tile_size(tensor.dtype))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{context} must expose a valid native tile") from error
    if len(tile_shape) != 2 or any(dimension <= 0 for dimension in tile_shape):
        raise ValueError(f"{context} has invalid native tile shape {tile_shape}")
    if page_size <= 0:
        raise ValueError(f"{context} has invalid native tile size {page_size}")

    shard_spec = getattr(memory_config, "shard_spec", None)
    try:
        shard_shape = tuple(int(dimension) for dimension in shard_spec.shape)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"{context} must expose a valid shard_spec.shape") from error
    if len(shard_shape) != 2 or any(dimension <= 0 for dimension in shard_shape):
        raise ValueError(f"{context} has invalid shard shape {shard_shape}")

    shard_rows, shard_columns = shard_shape
    tile_rows, tile_columns = tile_shape
    if shard_rows % tile_rows != 0 or shard_columns % tile_columns != 0:
        raise ValueError(
            f"{context} shard shape {shard_shape} must be divisible by native "
            f"tile shape {tile_shape}"
        )
    logical_shard_size_bytes = (
        (shard_rows // tile_rows) * (shard_columns // tile_columns) * page_size
    )
    return _TensorBackedDFBTensorProperties(
        tile_shape=tile_shape,
        page_size=page_size,
        logical_shard_size_bytes=logical_shard_size_bytes,
    )


def _validate_tensor_backed_dfb_range(
    properties: _TensorBackedDFBTensorProperties,
    *,
    byte_offset: int,
    byte_size: int,
    context: str,
) -> None:
    """Reject allocation-alignment slack that is not logical tensor data."""
    if byte_offset < 0 or byte_size <= 0:
        raise ValueError(f"{context} byte range must have positive size")
    if (
        byte_offset > _DFB_DESCRIPTOR_UINT32_MAX
        or byte_size > _DFB_DESCRIPTOR_UINT32_MAX
    ):
        raise ValueError(f"{context} byte range exceeds the uint32 descriptor fields")
    if byte_offset > _DFB_DESCRIPTOR_UINT32_MAX - byte_size:
        raise ValueError(
            f"{context} byte range end exceeds the uint32 descriptor fields"
        )
    if byte_offset % properties.page_size != 0:
        raise ValueError(
            f"{context} byte_offset must be aligned to the "
            f"{properties.page_size}-byte DFB page size"
        )
    if (
        byte_size > properties.logical_shard_size_bytes
        or byte_offset > properties.logical_shard_size_bytes - byte_size
    ):
        raise ValueError(
            f"{context} byte range [{byte_offset}, {byte_offset + byte_size}) "
            "exceeds logical per-shard size "
            f"{properties.logical_shard_size_bytes}"
        )


# Module-level counter for DFB index assignment in creation order
_cb_index_counter = 0


def _reset_cb_counter():
    """Reset the DFB index counter. Called at kernel start."""
    global _cb_index_counter
    _cb_index_counter = 0


def _next_cb_index():
    """Get next DFB index and increment counter."""
    global _cb_index_counter
    idx = _cb_index_counter
    _cb_index_counter += 1
    return idx


def get_cb_count():
    """Return number of DFBs allocated so far."""
    return _cb_index_counter


def _get_cb_tensor_type(cb_val):
    """Extract the tensor type from the MLIR DFB value type."""
    cb_type = ttl.CircularBufferType.maybe_downcast(cb_val.type)
    if cb_type is None:
        raise ValueError(f"Expected CircularBufferType, got {cb_val.type}")
    return RankedTensorType.get(cb_type.shape, cb_type.element_type)


@syntax("!ttl.cb")
class DataflowBuffer:
    """
    Dataflow buffer (DFB) for inter-thread communication.

    Dataflow buffers provide producer-consumer synchronization between
    compute and data movement threads.

    Can be instantiated via make_dataflow_buffer_like() in kernel body,
    then captured by thread closures. Methods generate TTL ops during compilation.
    """

    def __init__(
        self,
        tensor: Any,
        shape: Tuple[int, ...],
        block_count: int,
        dtype: Any = None,
        tile: Tuple[int, int] = (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE),
        tensor_backing: Any = None,
        byte_offset: int = 0,
        byte_size: Optional[int] = None,
        allocation_group: Optional[DFBAllocationGroup] = None,
    ):
        if len(shape) < 2:
            raise ValueError(f"DFB shape must have at least 2 dimensions, got {shape}")
        if block_count < 1 or block_count > 32:
            raise ValueError(f"block_count must be in range [1, 32], got {block_count}")
        normalized_tile = normalize_tile_dimensions(tile)
        # A buffer's dtype has one source: a backing tensor or an explicit
        # dtype. Supplying both is only valid when they resolve to the same type.
        if dtype is not None and getattr(tensor, "dtype", None) is not None:
            if _resolve_dfb_dtype(dtype) != _resolve_dfb_dtype(tensor.dtype):
                raise ValueError(
                    f"DataflowBuffer dtype {dtype!r} conflicts with backing "
                    f"tensor dtype {tensor.dtype!r}; pass only one"
                )

        self.tensor = tensor
        self.shape = shape
        self.block_count = block_count
        self._dtype = dtype
        self.tile = normalized_tile
        self.tensor_backing = tensor_backing
        self.byte_offset = byte_offset
        self.byte_size = byte_size
        if isinstance(allocation_group, DFBAllocationGroup):
            allocation_group = _bind_current_dfb_allocation_group(allocation_group)
        elif allocation_group is not None and not isinstance(
            allocation_group, _BoundDFBAllocationGroup
        ):
            raise TypeError(
                "allocation_group must be created by "
                "ttl.make_dfb_allocation_group(), got "
                f"{type(allocation_group).__name__}"
            )
        self.allocation_group = allocation_group
        self._cb_index = _next_cb_index()

    @property
    def dtype(self):
        if self._dtype is not None:
            return self._dtype
        if hasattr(self.tensor, "dtype"):
            return self.tensor.dtype
        raise ValueError(
            "DataflowBuffer has no dtype: build it with a tensor "
            "(make_dataflow_buffer_like) or an explicit dtype (make_dfb)"
        )

    def wait(ast_self: "DataflowBuffer") -> "TensorBlock":
        """
        Wait for data from the dataflow buffer (consumer acquire).

        Use in consumer threads to acquire data. Must be followed by pop()
        to signal consumption is complete.

        Returns:
            TensorBlock: The acquired data with DFB association.

        Example:
            block = dfb.wait()
            result = compute(block)
            block.pop()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_wait(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)

    def reserve(ast_self: "DataflowBuffer") -> "TensorBlock":
        """
        Reserve space in the dataflow buffer (producer acquire).

        Use in producer threads to acquire space for writing. Must be followed
        by push() to signal data is ready.

        Returns:
            TensorBlock: The reserved space with DFB association.

        Example:
            block = dfb.reserve()
            copy(stream[idx], block).wait()
            block.push()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_reserve(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)

    def publish(ast_self: "DataflowBuffer") -> None:
        """Publish the complete tensor-backed capacity to consumers."""
        cb_type = ttl.CircularBufferType.maybe_downcast(ast_self.type)
        if cb_type is None:
            raise ValueError(f"Expected CircularBufferType, got {ast_self.type}")
        total_tiles = math.prod(cb_type.shape) * cb_type.block_count
        published_type = RankedTensorType.get([1, total_tiles], cb_type.element_type)
        ttl.cb_reserve(published_type, ast_self, num_tiles=total_tiles)
        ttl.cb_push(ast_self, num_tiles=total_tiles)


# Backward-compatible alias. Existing user code using `ttl.CircularBuffer`
# continues to work; new code should prefer `DataflowBuffer`.
CircularBuffer = DataflowBuffer


@dataclass(frozen=True)
class PhysicalDFBConfig:
    """Runtime configuration for one physical dataflow buffer allocation.

    The final DFB index assignment determines this configuration. It is
    independent of whether the allocation serves user-declared,
    compiler-created, or multiple non-overlapping logical DFBs.
    `tile` is present only when the DFB element type is a TTCore tile.
    `allocation_nodes` distinguishes an unknown domain (`None`) from an exact,
    possibly empty, launch-node set.
    """

    dfb_index: int
    num_tiles: int
    data_format: str  # e.g., "bfloat16", "float32", "bfloat8_b"
    block_count: int
    page_size: int
    tile: Optional[Tuple[int, int]]
    storage_segments: Tuple["DFBStorageSegment", ...] = ()
    allocation_nodes: Optional[Tuple[Tuple[int, int], ...]] = None


@dataclass(frozen=True)
class DFBStorageSegment:
    """Storage selected for a physical DFB on an exact launch-node set."""

    nodes: Tuple[Tuple[int, int], ...]
    tensor_index: Optional[int] = None
    byte_offset: int = 0
    byte_size: Optional[int] = None

    @property
    def is_tensor_backed(self) -> bool:
        return self.tensor_index is not None


@dataclass(frozen=True)
class DFBConfigurationEpoch:
    """One physical DFB configuration installed for an execution epoch."""

    entry_reconfiguration_ordinal: Optional[int]
    config: PhysicalDFBConfig


@dataclass(frozen=True)
class DFBReconfigurationPlan:
    """Finalized boundary order and per-physical-DFB epoch configurations."""

    boundary_ordinals: Tuple[int, ...]
    dfb_epochs: Tuple[Tuple[DFBConfigurationEpoch, ...], ...]


def make_dataflow_buffer_like(
    tensor: Any,
    shape: Tuple[int, ...],
    block_count: int = 2,
    *,
    allocation_group: Optional[DFBAllocationGroup] = None,
) -> DataflowBuffer:
    """
    Create a dataflow buffer with properties derived from a tensor.

    Args:
        tensor: Tensor that determines the DFB's data type
        shape: Tile counts per dimension for wait/reserve operations
        block_count: Capacity multiplier (default 2 for double-buffering)
        allocation_group: Optional immutable identity requiring compiler-verified
            physical allocation sharing with the other group members

    Returns:
        DataflowBuffer for use in thread function closures
    """
    tile = (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)
    if hasattr(tensor, "get_tile"):
        tile = tuple(tensor.get_tile().tile_shape)
    return DataflowBuffer(
        tensor,
        shape,
        block_count,
        tile=tile,
        allocation_group=allocation_group,
    )


def make_tensor_backed_dfb(
    tensor: Any,
    shape: Tuple[int, ...],
    *,
    block_count: int = 1,
    byte_offset: int = 0,
    allocation_group: Optional[DFBAllocationGroup] = None,
) -> DataflowBuffer:
    """Bind a DFB's complete capacity to a sharded L1 tensor byte range.

    ``allocation_group`` requires compiler-verified physical allocation sharing
    with the other group members. Tensor-backed group members must retain an
    identical DFB capacity descriptor.
    """
    from .dtype_utils import is_ttnn_tensor

    if not is_ttnn_tensor(tensor):
        raise TypeError("tensor-backed DFB storage requires a TTNN tensor")
    if not isinstance(byte_offset, int) or isinstance(byte_offset, bool):
        raise TypeError("byte_offset must be an integer")
    if byte_offset < 0:
        raise ValueError("byte_offset must be non-negative")
    if not isinstance(block_count, int) or isinstance(block_count, bool):
        raise TypeError("block_count must be an integer")
    if block_count < 1 or block_count > 32:
        raise ValueError(f"block_count must be in range [1, 32], got {block_count}")
    if len(shape) < 2 or any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
        for dimension in shape
    ):
        raise ValueError(f"DFB shape dimensions must be positive integers, got {shape}")

    context = "tensor-backed DFB storage"
    properties = _validate_tensor_backed_dfb_tensor(tensor, context=context)
    byte_size = math.prod(shape) * block_count * properties.page_size
    _validate_tensor_backed_dfb_range(
        properties,
        byte_offset=byte_offset,
        byte_size=byte_size,
        context=context,
    )

    return DataflowBuffer(
        tensor,
        shape,
        block_count,
        tile=properties.tile_shape,
        tensor_backing=tensor,
        byte_offset=byte_offset,
        byte_size=byte_size,
        allocation_group=allocation_group,
    )


def _resolve_dfb_dtype(dtype: Any):
    """Resolve a ``make_dfb`` dtype argument to a ttnn.DataType.

    Accepts a data-format name string ("bf16", "float32", ...), a ttnn
    dtype (returned as-is), or a torch dtype.
    """
    from .dtype_utils import format_name_to_ttnn_dtype, torch_dtype_to_ttnn_datatype

    if isinstance(dtype, str):
        return format_name_to_ttnn_dtype(dtype)
    if hasattr(dtype, "name"):  # already a ttnn.DataType enum
        return dtype
    return torch_dtype_to_ttnn_datatype(dtype)


def make_dfb(
    dtype: Any,
    shape: Tuple[int, ...],
    block_count: int = 2,
    tile: Tuple[int, int] = (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE),
    *,
    allocation_group: Optional[DFBAllocationGroup] = None,
) -> DataflowBuffer:
    """
    Create a dataflow buffer from an explicit dtype, with no backing tensor.

    Unlike make_dataflow_buffer_like, no dummy tensor is needed.

    Args:
        dtype: Element data type. Accepts a data-format name string
            ("bf16", "float32", ...), a ttnn dtype, or a torch dtype.
        shape: Tile counts per dimension for wait/reserve operations
        block_count: Capacity multiplier (default 2 for double-buffering)
        tile: Physical tile dimensions. tt-metal supports heights 1, 2, 4, 8,
            16, or 32 and widths 16 or 32.
        allocation_group: Optional immutable identity requiring compiler-verified
            physical allocation sharing with the other group members

    Returns:
        DataflowBuffer for use in thread function closures
    """
    return DataflowBuffer(
        tensor=None,
        shape=shape,
        block_count=block_count,
        dtype=_resolve_dfb_dtype(dtype),
        tile=tile,
        allocation_group=allocation_group,
    )
