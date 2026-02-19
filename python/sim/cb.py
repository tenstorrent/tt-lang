# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CircularBuffer: High-level interface for tensor-aware circular buffers.

This module provides a CircularBuffer class that wraps the low-level CBAPI
and provides a convenient interface for managing circular buffers with
tensor data. It handles CB allocation, configuration, and provides tensor-aware
operations.
"""

from typing import List, Optional, Tuple

import torch

from .block import Block
from .cbapi import CBAPI
from .constants import TILE_SHAPE
from .stats import record_cb_reserve, record_cb_wait
from .ttnnsim import Tensor
from .typedefs import CBID, Shape, Size


# TODO: Should this class now be private?
class CircularBuffer:
    """
    High-level circular buffer interface for tensor operations.

    This class provides a convenient wrapper around the low-level CBAPI,
    handling CB allocation and providing tensor-aware operations.

    The CircularBuffer manages a fixed-size circular buffer with space for
    a configurable number of tiles. Operations like wait() and reserve()
    work with a fixed number of tiles determined by the shape parameter.

    Example:
        cb = CircularBuffer(shape=(2, 3), buffer_factor=2)

        # Producer workflow
        write_view = cb.reserve()  # Reserve space for 6 tiles
        # ... write data to write_view ...
        write_view.push()  # Make data visible

        # Consumer workflow
        read_view = cb.wait()  # Wait for 6 tiles
        # ... read data from read_view ...
        read_view.pop()  # Free consumed tiles
    """

    def __init__(
        self,
        element: Tensor,
        shape: Shape,
        buffer_factor: Size = 2,
        api: Optional[CBAPI] = None,
    ):
        """
        Initialize a CircularBuffer.

        Args:
            element: A tensor used to determine the dtype for zero-initialized tensors in reserved blocks
            shape: Tuple of (rows, cols) specifying the tile shape for wait/reserve operations
            buffer_factor: Multiplier for total buffer capacity (capacity = shape[0] * shape[1] * buffer_factor)
            api: Optional CBAPI instance to use. If None, uses the shared default instance.

        Raises:
            ValueError: If shape or buffer_factor are invalid
            RuntimeError: If CB allocation fails
        """
        if len(shape) != 2:
            raise ValueError(f"Shape must be a 2-tuple, got {shape}")

        self.element = element
        self._shape = shape
        self._buffer_factor = buffer_factor

        # Store API instance (may be None)
        self._api: Optional[CBAPI] = api

        # Track pending blocks for state machine completion
        # At most one pending reserved block and one pending waited block at a time
        self._pending_reserved_block: Optional[Block] = None
        self._pending_waited_block: Optional[Block] = None

        # Calculate total capacity in tiles
        self._tiles_per_operation = shape[0] * shape[1]
        self._capacity_tiles = self._tiles_per_operation * buffer_factor

        # Only allocate and configure if API is provided
        # If None, this will be done when the CB is copied by Program
        if self._api is not None:
            self._cb_id: Optional[CBID] = self._api.allocate_cb_id()
            self._api.host_configure_cb(self._cb_id, self._capacity_tiles, self._shape)
            # Reset the buffer to initialize with zero entries
            self._api.host_reset_cb(self._cb_id)
        else:
            self._cb_id: Optional[CBID] = None  # Placeholder until properly initialized

    def _ensure_initialized(self) -> Tuple[CBAPI, CBID]:
        """Verify that the CircularBuffer has been properly initialized with an API.

        Returns:
            Tuple of (api, cb_id) for use in operations

        Raises:
            RuntimeError: If the CB was not initialized with an API instance
        """
        if self._api is None or self._cb_id is None:
            raise RuntimeError(
                "CircularBuffer was not properly initialized with a CBAPI instance. "
                "This likely means it was created outside of a kernel context. "
                "CircularBuffers must be created within @ttl.kernel decorated functions."
            )
        return self._api, self._cb_id

    def wait(self) -> Block:
        """Wait for data to be available and return a read view.

        This method blocks until the required number of tiles (as specified by
        the shape parameter) are available for reading. It returns a Block
        that provides access to the available data.

        Usage:
            blk = cb.wait()
            data = blk[0]
            blk.pop()  # manual pop required

        Returns:
            Block providing read access to the available tiles

        Raises:
            CBTimeoutError: If the wait times out
            CBContractError: If called incorrectly (e.g., multiple concurrent waits)
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()

        # Enforce: at most one pending wait() operation at a time
        if self._pending_waited_block is not None:
            raise RuntimeError(
                "Cannot call wait() again before pop(): "
                "CircularBuffer already has a pending waited block. "
                "You must call pop() before calling wait() again."
            )

        # Block if data not available
        from .greenlet_scheduler import block_if_needed

        block_if_needed(self, "wait")

        api.cb_wait_front(cb_id, self._tiles_per_operation)
        block = api.get_read_ptr(cb_id)
        block.cb = self  # Set CB reference for context manager support
        self._pending_waited_block = block

        # Record wait statistics
        record_cb_wait(self, self._tiles_per_operation)

        return block

    def can_wait(self) -> bool:
        """
        Check if wait() can proceed without blocking.

        Returns:
            True if sufficient data is available for wait(), False otherwise

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        stats = api.cb_stats(cb_id)
        return stats.visible >= self._tiles_per_operation

    def reserve(self) -> Block:
        """
        Reserve space for writing and return a write view.

        This method blocks until there is sufficient space to write the required
        number of tiles (as specified by the shape parameter). It returns a Block
        that provides access to the reserved space.

        The reserved block is automatically initialized with zero tensors using
        TILE_SHAPE dimensions and the element's dtype before being returned.

        Usage:
            blk = cb.reserve()
            blk.store(data)
            blk.push()  # manual push required

        Returns:
            Block providing write access to the reserved space

        Raises:
            CBTimeoutError: If the reservation times out
            CBContractError: If called incorrectly (e.g., multiple concurrent reserves)
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()

        # Enforce: at most one pending reserve() operation at a time
        if self._pending_reserved_block is not None:
            raise RuntimeError(
                "Cannot call reserve() again before push(): "
                "CircularBuffer already has a pending reserved block. "
                "You must call push() before calling reserve() again."
            )

        # Block if space not available
        from .greenlet_scheduler import block_if_needed

        block_if_needed(self, "reserve")

        api.cb_reserve_back(cb_id, self._tiles_per_operation)
        block = api.get_write_ptr(cb_id)
        block.cb = self  # Set CB reference for context manager support

        # Initialize the reserved block with zero tensors
        zero_tensor = Tensor(torch.zeros(TILE_SHAPE, dtype=self.element.dtype))
        for i in range(len(block)):
            block.write_slot(i, zero_tensor)

        self._pending_reserved_block = block

        # Record reserve statistics
        record_cb_reserve(self, self._tiles_per_operation)

        return block

    def can_reserve(self) -> bool:
        """
        Check if reserve() can proceed without blocking.

        Returns:
            True if sufficient space is available for reserve(), False otherwise

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        stats = api.cb_stats(cb_id)
        return stats.free >= self._tiles_per_operation

    def push_block(self) -> None:
        """
        Finalize a write operation, making reserved data visible to consumers.

        This method must be called after reserve() and writing data to the
        returned Block. It advances the CB pointers and makes the written
        data available for consumers to read via wait().

        Raises:
            CBContractError: If called without a prior reserve() or if the
                           push amount exceeds what was reserved
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        # Update state machine for the pending reserved block
        if self._pending_reserved_block is not None:
            self._pending_reserved_block.mark_push_complete()
            self._pending_reserved_block = None

        api, cb_id = self._ensure_initialized()
        api.cb_push_back(cb_id, self._tiles_per_operation)

    def pop_block(self) -> None:
        """
        Finalize a read operation, freeing consumed data.

        This method must be called after wait() and reading data from the
        returned Block. It advances the CB pointers and frees the consumed
        tiles, making space available for producers.

        After calling pop(), the Block returned by the corresponding wait()
        points to stale data and should not be accessed.

        Raises:
            CBContractError: If called without a prior wait() or if the
                           pop amount exceeds what is visible
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        # Update state machine for the pending waited block
        if self._pending_waited_block is not None:
            self._pending_waited_block.mark_pop_complete()
            self._pending_waited_block = None

        api, cb_id = self._ensure_initialized()
        api.cb_pop_front(cb_id, self._tiles_per_operation)

    @property
    def shape(self) -> Tuple[Size, Size]:
        """Get the shape (in tiles) for wait/reserve operations."""
        return self._shape

    @property
    def capacity_tiles(self) -> Size:
        """Get the total capacity of the buffer in tiles."""
        return self._capacity_tiles

    @property
    def buffer_factor(self) -> Size:
        """Get the buffer factor (capacity multiplier)."""
        return self._buffer_factor

    @property
    def cb_id(self) -> Optional[CBID]:
        """Get the internal CB ID (for debugging/advanced use)."""
        return self._cb_id

    def stats(self):
        """Get current buffer statistics.

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        return api.cb_stats(cb_id)

    def reset(self) -> None:
        """Reset the circular buffer to initial state.

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        api.host_reset_cb(cb_id)

    def validate_no_pending_blocks(self) -> None:
        """Validate that there are no pending blocks.

        This should be called at the end of kernel execution to ensure
        all blocks have been properly completed through push() or pop().

        Raises:
            RuntimeError: If there are any pending blocks
        """
        errors: List[str] = []

        if self._pending_reserved_block is not None:
            block = self._pending_reserved_block
            errors.append(
                f"Pending reserved block: Block(acquisition={block.acquisition.name}, "
                f"thread={block.thread_type.name}, access={block.access_state.name}, "
                f"expected_ops={[op.name for op in block.expected_ops]}). "
                f"Did you forget to call push()?"
            )

        if self._pending_waited_block is not None:
            block = self._pending_waited_block
            errors.append(
                f"Pending waited block: Block(acquisition={block.acquisition.name}, "
                f"thread={block.thread_type.name}, access={block.access_state.name}, "
                f"expected_ops={[op.name for op in block.expected_ops]}). "
                f"Did you forget to call pop()?"
            )

        if errors:
            raise RuntimeError(
                f"CircularBuffer {self} has incomplete blocks at end of execution:\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

    def __repr__(self) -> str:
        return (
            f"CircularBuffer(cb_id={self._cb_id}, shape={self._shape}, "
            f"capacity_tiles={self._capacity_tiles}, buffer_factor={self._buffer_factor})"
        )


def make_circular_buffer_like(
    element: Tensor,
    shape: Shape,
    buffer_factor: Size = 2,
) -> CircularBuffer:
    """
    Create a CircularBuffer with the same dtype as the element.

    Args:
        element: A tensor used to determine the CircularBuffer's dtype
        shape: Tuple of (rows, cols) specifying the tile shape for wait/reserve operations
        buffer_factor: Multiplier for total buffer capacity (capacity = shape[0] * shape[1] * buffer_factor)

    Returns:
        A CircularBuffer with dtype matching the element

    Example:
        x = ttnn.zeros((32, 32), dtype=ttnn.float32)
        x_cb = make_circular_buffer_like(x, shape=(2, 2), buffer_factor=2)
    """
    return CircularBuffer(element=element, shape=shape, buffer_factor=buffer_factor)
