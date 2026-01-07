# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Circular buffer operations for inter-thread communication."""

from typing import Tuple, Any

from ttmlir.ir import *

from .dialects import ttl
from ._src.ttl_ast import syntax

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
    """Get current total CB count (kernel-scope)."""
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

    Can be instantiated via make_circular_buffer_like() in kernel body,
    then captured by thread closures. Methods generate TTL ops during compilation.
    """

    def __init__(
        self,
        tensor: Any,
        shape: Tuple[int, int],
        buffer_factor: int,
    ):
        if len(shape) != 2:
            raise ValueError(f"shape must be a 2-tuple, got {shape}")
        if buffer_factor < 1 or buffer_factor > 32:
            raise ValueError(
                f"buffer_factor must be in range [1, 32], got {buffer_factor}"
            )

        self.tensor = tensor
        self.shape = shape
        self.buffer_factor = buffer_factor
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
            shard = cb.wait()
            result = compute(shard)
            cb.pop()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_wait(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)

    def pop(ast_self: "CircularBuffer") -> None:
        """
        Signal that data has been consumed (consumer release).

        Use in consumer threads after wait() to signal that data has been
        consumed and space is available for producers.

        Example:
            shard = cb.wait()
            result = compute(shard)
            cb.pop()  # Signal consumption complete
        """
        ttl.cb_pop(ast_self)

    def reserve(ast_self: "CircularBuffer") -> "TensorBlock":
        """
        Reserve space in the circular buffer (producer acquire).

        Use in producer threads to acquire space for writing. Must be followed
        by push() to signal data is ready.

        Returns:
            TensorBlock: The reserved space with CB association.

        Example:
            cb.reserve()
            copy(stream[idx], cb).wait()
            cb.push()
        """
        tensor_type = _get_cb_tensor_type(ast_self)
        tensor = ttl.cb_reserve(tensor_type, ast_self)
        return ttl.attach_cb(tensor.type, tensor, ast_self)

    def push(ast_self: "CircularBuffer") -> None:
        """
        Signal that data is ready in the circular buffer (producer release).

        Use in producer threads after reserve() to signal that data has been
        written and is ready for consumers.

        Example:
            shard = cb.reserve()
            copy(stream[idx], shard).wait()
            cb.push()  # Signal data ready
        """
        ttl.cb_push(ast_self)


def make_circular_buffer_like(
    tensor: Any,
    shape: Tuple[int, int],
    buffer_factor: int = 2,
) -> CircularBuffer:
    """
    Create a circular buffer with properties derived from a tensor.

    Args:
        tensor: Tensor that determines the CB's data type
        shape: (rows, cols) in tiles for wait/reserve operations
        buffer_factor: Capacity multiplier (default 2 for double-buffering)

    Returns:
        CircularBuffer for use in thread function closures
    """
    return CircularBuffer(tensor, shape, buffer_factor)
