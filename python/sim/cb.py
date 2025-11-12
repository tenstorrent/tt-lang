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

from typing import Tuple, Optional
import torch
from threading import Lock

from .cbapi import CBAPI
from .ringview import RingView
from .typedefs import CBID, Size
from .constants import MAX_CBS


# TODO: Perhaps make CircularBuffer generic as well? Currently supports
# torch.Tensor only.
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
        cb.push()  # Make data visible

        # Consumer workflow
        read_view = cb.wait()  # Wait for 6 tiles
        # ... read data from read_view ...
        cb.pop()  # Free consumed tiles
    """

    # Class-level CB allocator for default API instance
    _cb_allocator_lock = Lock()
    _next_cb_id = 0
    _default_api = CBAPI[torch.Tensor]()

    @classmethod
    def _allocate_cb_id(cls) -> CBID:
        """Allocate a unique CB ID. Thread-safe."""
        with cls._cb_allocator_lock:
            cb_id = cls._next_cb_id
            cls._next_cb_id += 1
            if cls._next_cb_id >= MAX_CBS:
                raise RuntimeError("Maximum number of circular buffers exceeded")
            return cb_id

    def __init__(
        self,
        shape: Tuple[Size, Size],
        buffer_factor: Size = 2,
        api: Optional[CBAPI[torch.Tensor]] = None,
    ):
        """
        Initialize a CircularBuffer.

        Args:
            shape: Tuple of (rows, cols) specifying the tile shape for wait/reserve operations
            buffer_factor: Multiplier for total buffer capacity (capacity = shape[0] * shape[1] * buffer_factor)
            api: Optional CBAPI instance to use. If None, uses the shared default instance.

        Raises:
            ValueError: If shape or buffer_factor are invalid
            RuntimeError: If CB allocation fails
        """
        if len(shape) != 2:
            raise ValueError(f"Shape must be a 2-tuple, got {shape}")

        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Shape dimensions must be positive, got {shape}")

        if buffer_factor <= 0:
            raise ValueError(f"Buffer factor must be positive, got {buffer_factor}")

        self._shape = shape
        self._buffer_factor = buffer_factor

        # Use provided API instance or default
        self._api = api if api is not None else self._default_api

        # Calculate total capacity in tiles
        self._tiles_per_operation = shape[0] * shape[1]
        self._capacity_tiles = self._tiles_per_operation * buffer_factor

        # Allocate and configure CB
        self._cb_id = self._allocate_cb_id()
        self._api.host_configure_cb(self._cb_id, self._capacity_tiles)

    def wait(self) -> RingView[torch.Tensor]:
        """
        Wait for data to be available and return a read view.

        This method blocks until the required number of tiles (as specified by
        the shape parameter) are available for reading. It returns a RingView
        that provides access to the available data.

        Returns:
            RingView object providing read access to the available tiles

        Raises:
            CBTimeoutError: If the wait times out
            CBContractError: If called incorrectly (e.g., multiple concurrent waits)
        """
        self._api.cb_wait_front(self._cb_id, self._tiles_per_operation)
        return self._api.get_read_ptr(self._cb_id)

    def reserve(self) -> RingView[torch.Tensor]:
        """
        Reserve space for writing and return a write view.

        This method blocks until there is sufficient space to write the required
        number of tiles (as specified by the shape parameter). It returns a RingView
        that provides access to the reserved space.

        Returns:
            RingView object providing write access to the reserved space

        Raises:
            CBTimeoutError: If the reservation times out
            CBContractError: If called incorrectly (e.g., multiple concurrent reserves)
        """
        self._api.cb_reserve_back(self._cb_id, self._tiles_per_operation)
        return self._api.get_write_ptr(self._cb_id)

    def push(self) -> None:
        """
        Finalize a write operation, making reserved data visible to consumers.

        This method must be called after reserve() and writing data to the
        returned RingView. It advances the CB pointers and makes the written
        data available for consumers to read via wait().

        Raises:
            CBContractError: If called without a prior reserve() or if the
                           push amount exceeds what was reserved
        """
        self._api.cb_push_back(self._cb_id, self._tiles_per_operation)

    def pop(self) -> None:
        """
        Finalize a read operation, freeing consumed data.

        This method must be called after wait() and reading data from the
        returned RingView. It advances the CB pointers and frees the consumed
        tiles, making space available for producers.

        After calling pop(), the RingView returned by the corresponding wait()
        points to stale data and should not be accessed.

        Raises:
            CBContractError: If called without a prior wait() or if the
                           pop amount exceeds what is visible
        """
        self._api.cb_pop_front(self._cb_id, self._tiles_per_operation)

    @property
    def shape(self) -> Tuple[Size, Size]:
        """Get the shape (in tiles) for wait/reserve operations."""
        return self._shape

    @property
    def capacity_tiles(self) -> int:
        """Get the total capacity of the buffer in tiles."""
        return self._capacity_tiles

    @property
    def buffer_factor(self) -> Size:
        """Get the buffer factor (capacity multiplier)."""
        return self._buffer_factor

    @property
    def cb_id(self) -> CBID:
        """Get the internal CB ID (for debugging/advanced use)."""
        return self._cb_id

    def stats(self):
        """Get current buffer statistics."""
        return self._api.cb_stats(self._cb_id)

    def reset(self) -> None:
        """Reset the circular buffer to initial state."""
        self._api.host_reset_cb(self._cb_id)

    def __repr__(self) -> str:
        return (
            f"CircularBuffer(cb_id={self._cb_id}, shape={self._shape}, "
            f"capacity_tiles={self._capacity_tiles}, buffer_factor={self._buffer_factor})"
        )
