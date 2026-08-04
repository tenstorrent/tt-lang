# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dataflow buffer (DFB) operations for inter-thread communication."""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ttl.ir import *

from ._src.ttl_ast import syntax
from ttl.dialects import ttl

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
        tile: Tuple[int, int] = (32, 32),
    ):
        if len(shape) < 2:
            raise ValueError(f"DFB shape must have at least 2 dimensions, got {shape}")
        if block_count < 1 or block_count > 32:
            raise ValueError(f"block_count must be in range [1, 32], got {block_count}")
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
        self.tile = tuple(tile)
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
    """

    dfb_index: int
    num_tiles: int
    data_format: str  # e.g., "bfloat16", "float32", "float16"
    block_count: int
    page_size: int
    tile: Optional[Tuple[int, int]]


def make_dataflow_buffer_like(
    tensor: Any,
    shape: Tuple[int, ...],
    block_count: int = 2,
) -> DataflowBuffer:
    """
    Create a dataflow buffer with properties derived from a tensor.

    Args:
        tensor: Tensor that determines the DFB's data type
        shape: Tile counts per dimension for wait/reserve operations
        block_count: Capacity multiplier (default 2 for double-buffering)

    Returns:
        DataflowBuffer for use in thread function closures
    """
    tile = (32, 32)
    if hasattr(tensor, "get_tile"):
        tile = tuple(tensor.get_tile().tile_shape)
    return DataflowBuffer(tensor, shape, block_count, tile=tile)


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
    tile: Tuple[int, int] = (32, 32),
) -> DataflowBuffer:
    """
    Create a dataflow buffer from an explicit dtype, with no backing tensor.

    Unlike make_dataflow_buffer_like, no dummy tensor is needed.

    Args:
        dtype: Element data type. Accepts a data-format name string
            ("bf16", "float32", ...), a ttnn dtype, or a torch dtype.
        shape: Tile counts per dimension for wait/reserve operations
        block_count: Capacity multiplier (default 2 for double-buffering)

    Returns:
        DataflowBuffer for use in thread function closures
    """
    return DataflowBuffer(
        tensor=None,
        shape=shape,
        block_count=block_count,
        dtype=_resolve_dfb_dtype(dtype),
        tile=tile,
    )
