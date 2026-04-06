# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Circular buffer operations for inter-thread communication."""

from typing import Any, Tuple

from ttl.ir import *

from ._src.ttl_ast import syntax
from ttl.dialects import ttl

# Module-level counter for CB index assignment in creation order
_cb_index_counter = 0


def _reset_cb_counter():
    """Reset the CB index counter. Called at kernel start."""
    global _cb_index_counter
    _cb_index_counter = 0


def _next_cb_index():
    """Get next CB index and increment counter."""
    global _cb_index_counter
    idx = _cb_index_counter
    _cb_index_counter += 1
    return idx


def get_cb_count():
    """Return number of CBs allocated so far."""
    return _cb_index_counter


def _get_cb_tensor_type(cb_val):
    """Extract the tensor type from a TTL CB type."""
    cb_type = ttl.CircularBufferType.maybe_downcast(cb_val.type)
    if cb_type is None:
        raise ValueError(f"Expected CircularBufferType, got {cb_val.type}")
    return RankedTensorType.get(cb_type.shape, cb_type.element_type)


@syntax("!ttl.cb")
class CircularBuffer:
    """
    Circular buffer for inter-thread communication.

    Circular buffers provide producer-consumer synchronization between
    compute and data movement threads.

    Can be instantiated via make_dataflow_buffer_like() in kernel body,
    then captured by thread closures. Methods generate TTL ops during compilation.
    """

    def __init__(
        self,
        tensor: Any,
        shape: Tuple[int, ...],
        block_count: int,
    ):
        if len(shape) < 2:
            raise ValueError(f"CB shape must have at least 2 dimensions, got {shape}")
        if block_count < 1 or block_count > 32:
            raise ValueError(f"block_count must be in range [1, 32], got {block_count}")

        self.tensor = tensor
        self.shape = shape
        self.block_count = block_count
        self._cb_index = _next_cb_index()

    @property
    def dtype(self):
        if hasattr(self.tensor, "dtype"):
            return self.tensor.dtype
        raise ValueError("tensor has no dtype attribute")

    def wait(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Wait for data from the circular buffer (consumer acquire).

        Use in consumer threads to acquire data. Must be followed by pop()
        to signal consumption is complete.

        Returns:
            TensorBlock: The acquired data with CB association.

        Example:
            block = cb.wait()
            result = compute(block)
            block.pop()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_wait(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)

    def reserve(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Reserve space in the circular buffer (producer acquire).

        Use in producer threads to acquire space for writing. Must be followed
        by push() to signal data is ready.

        Returns:
            TensorBlock: The reserved space with CB association.

        Example:
            block = cb.reserve()
            copy(stream[idx], block).wait()
            block.push()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_reserve(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)


def make_dataflow_buffer_like(
    tensor: Any,
    shape: Tuple[int, ...],
    block_count: int = 2,
) -> CircularBuffer:
    """
    Create a circular buffer with properties derived from a tensor.

    Args:
        tensor: Tensor that determines the CB's data type
        shape: Tile counts per dimension for wait/reserve operations
        block_count: Capacity multiplier (default 2 for double-buffering)

    Returns:
        CircularBuffer for use in thread function closures
    """
    return CircularBuffer(tensor, shape, block_count)
