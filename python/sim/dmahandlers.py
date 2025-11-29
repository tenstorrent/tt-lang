# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DMA transfer handlers using a registry-based strategy pattern.

Each handler implements validate() and transfer() for a specific (src_type, dst_type) pair.
New transfer types can be added by creating a new handler and decorating it with
@register_dma_handler.
"""

from typing import Any, Dict, Protocol, Tuple, Type, Deque, List, Union, TypedDict
from collections import deque
import threading
import time
import torch
from . import torch_utils as tu
from .ringview import RingView
from .constants import TILE_SHAPE, DMA_MULTICAST_TIMEOUT
from .typedefs import MulticastAddress, Count

# DMA endpoint types - these are the valid types for DMA transfers
# To add a new endpoint type, add it to this Union and implement a handler for it
DMAEndpoint = Union[torch.Tensor, RingView[torch.Tensor], MulticastAddress]
# Type of a DMA endpoint class (derived automatically from DMAEndpoint)
DMAEndpointType = Type[DMAEndpoint]


# Global multicast state for simulating NoC multicast communication
# For each multicast address we keep a small structure with:
# - queue: deque of (data, remaining_receiver_count)
# - event: threading.Event set when queue is non-empty
# - lock: threading.Lock to guard queue and receiver count updates
# In a real implementation this would be handled by NoC hardware.
class _MulticastEntry(TypedDict):
    queue: Deque[Tuple[List[torch.Tensor], Count]]
    event: threading.Event
    lock: threading.Lock


_multicast_buffer: Dict[MulticastAddress, _MulticastEntry] = {}
# Lock protecting creation of per-address entries in _multicast_buffer.
# This ensures all threads agree on the same entry object (and its lock)
# and avoids races where two threads create different entry dicts for
# the same multicast address.
_multicast_registry_lock = threading.Lock()


class DMATransferHandler(Protocol):
    """Protocol for DMA transfer handlers."""

    def validate(self, src: Any, dst: Any) -> None:
        """
        Validate that the transfer can be performed.

        Args:
            src: Source object
            dst: Destination object

        Raises:
            ValueError: If the transfer is not valid (shape mismatch, etc.)
        """
        ...

    def transfer(self, src: Any, dst: Any) -> None:
        """
        Perform the actual data transfer.

        Args:
            src: Source object
            dst: Destination object

        Raises:
            ValueError: If the transfer fails
        """
        ...


# Global handler registry: (src_type, dst_type) -> handler instance
handler_registry: Dict[Tuple[DMAEndpointType, DMAEndpointType], DMATransferHandler] = {}


def register_dma_handler(src_type: DMAEndpointType, dst_type: DMAEndpointType):
    """
    Decorator to register a DMA transfer handler for a specific (src_type, dst_type) pair.

    Args:
        src_type: Source type class (must be a valid DMA endpoint type)
        dst_type: Destination type class (must be a valid DMA endpoint type)

    Returns:
        Decorator function

    Example:
        @register_dma_handler(torch.Tensor, RingView)
        class TensorToRingViewHandler:
            def validate(self, src, dst): ...
            def transfer(self, src, dst): ...
    """

    def decorator(handler_cls: Type[DMATransferHandler]):
        handler_registry[(src_type, dst_type)] = handler_cls()
        return handler_cls

    return decorator


@register_dma_handler(torch.Tensor, RingView)
class TensorToRingViewHandler:
    """Handler for Tensor → RingView transfers."""

    def validate(self, src: torch.Tensor, dst: RingView[torch.Tensor]) -> None:
        """Validate tensor to RingView transfer."""
        if len(src.shape) != 2:
            raise ValueError(
                f"DMA only supports 2D tensors, got {len(src.shape)}D tensor with shape {src.shape}"
            )

        num_tiles = tu.tile_count(src.shape, TILE_SHAPE)
        expected_tiles = len(dst)

        if num_tiles != expected_tiles:
            raise ValueError(
                f"Tensor contains {num_tiles} tiles but RingView has {expected_tiles} slots"
            )

    def transfer(self, src: torch.Tensor, dst: RingView[torch.Tensor]) -> None:
        """Transfer tensor data to RingView by splitting into tiles."""
        num_tiles = tu.tile_count(src.shape, TILE_SHAPE)
        width_tiles = src.shape[1] // TILE_SHAPE[1]

        # Extract tiles in row-major order
        for tile_idx in range(num_tiles):
            # Convert linear index to 2D tile coordinates
            h_tile = tile_idx // width_tiles
            w_tile = tile_idx % width_tiles

            # Calculate tensor slice coordinates
            start_row = h_tile * TILE_SHAPE[0]
            end_row = (h_tile + 1) * TILE_SHAPE[0]
            start_col = w_tile * TILE_SHAPE[1]
            end_col = (w_tile + 1) * TILE_SHAPE[1]

            tile = src[start_row:end_row, start_col:end_col]
            dst[tile_idx] = tile


@register_dma_handler(RingView, torch.Tensor)
class RingViewToTensorHandler:
    """Handler for RingView → Tensor transfers."""

    def validate(self, src: RingView[torch.Tensor], dst: torch.Tensor) -> None:
        """Validate RingView to tensor transfer."""
        if len(dst.shape) != 2:
            raise ValueError(
                f"DMA only supports 2D tensors, got {len(dst.shape)}D tensor with shape {dst.shape}"
            )

        dst_tiles = tu.tile_count(dst.shape, TILE_SHAPE)
        if len(src) != dst_tiles:
            raise ValueError(f"Expected {len(src)} tiles but found {dst_tiles}")

    def transfer(self, src: RingView[torch.Tensor], dst: torch.Tensor) -> None:
        """Transfer RingView data to tensor by combining tiles."""
        dst_tiles = tu.tile_count(dst.shape, TILE_SHAPE)
        width_tiles = dst.shape[1] // TILE_SHAPE[1]

        # Reconstruct tensor by placing tiles in their proper 2D positions
        for tile_idx in range(dst_tiles):
            # Convert linear index to 2D tile coordinates
            h_tile = tile_idx // width_tiles
            w_tile = tile_idx % width_tiles

            start_row = h_tile * TILE_SHAPE[0]
            end_row = (h_tile + 1) * TILE_SHAPE[0]
            start_col = w_tile * TILE_SHAPE[1]
            end_col = (w_tile + 1) * TILE_SHAPE[1]

            tile = src[tile_idx]
            dst[start_row:end_row, start_col:end_col] = tile


@register_dma_handler(RingView, MulticastAddress)
class RingViewToMulticastHandler:
    """Handler for RingView → MulticastAddress (multicast send)."""

    def validate(self, src: RingView[torch.Tensor], dst: MulticastAddress) -> None:
        """Validate multicast send - no specific validation needed."""
        pass

    def transfer(self, src: RingView[torch.Tensor], dst: MulticastAddress) -> None:
        """Multicast send: store data in shared buffer accessible by all cores."""
        src_data = [src[i] for i in range(len(src))]
        # Initialize per-address state atomically so all threads see the
        # same entry (and therefore the same per-entry lock).
        with _multicast_registry_lock:
            entry = _multicast_buffer.get(dst)
            if entry is None:
                new_entry: _MulticastEntry = {
                    "queue": deque(),
                    "event": threading.Event(),
                    "lock": threading.Lock(),
                }
                _multicast_buffer[dst] = new_entry
                entry = new_entry

        # Calculate number of receivers (all cores except the sender)
        num_receivers = len(dst.core_indices) - 1

        # Add to the queue for this multicast address with receiver count
        # and notify any waiting receivers via event
        with entry["lock"]:
            entry["queue"].append((src_data, num_receivers))
            # Signal that data is available
            entry["event"].set()


@register_dma_handler(MulticastAddress, RingView)
class MulticastToRingViewHandler:
    """Handler for MulticastAddress → RingView (multicast receive)."""

    def validate(self, src: MulticastAddress, dst: RingView[torch.Tensor]) -> None:
        """Validate multicast receive - validation happens during transfer when data is available."""
        pass

    def transfer(self, src: MulticastAddress, dst: RingView[torch.Tensor]) -> None:
        """Multicast receive: retrieve data from shared multicast buffer."""
        # Use an event to wait for data instead of polling. This reduces CPU
        # usage and provides a cleaner synchronization primitive for tests.
        start_time = time.time()

        # Ensure entry exists atomically so we can safely access event/lock.
        with _multicast_registry_lock:
            entry = _multicast_buffer.get(src)
            if entry is None:
                new_entry: _MulticastEntry = {
                    "queue": deque(),
                    "event": threading.Event(),
                    "lock": threading.Lock(),
                }
                _multicast_buffer[src] = new_entry
                entry = new_entry
        event: threading.Event = entry["event"]
        queue: Deque[Tuple[List[torch.Tensor], Count]] = entry["queue"]
        lock: threading.Lock = entry["lock"]

        while True:
            # Compute remaining timeout
            elapsed = time.time() - start_time
            remaining = DMA_MULTICAST_TIMEOUT - elapsed
            if remaining <= 0:
                raise TimeoutError(
                    f"Timeout waiting for multicast data. "
                    f"The sender may not have called dma(ringview, mcast_addr).wait() "
                    f"or there may be a deadlock."
                )

            # Wait until signaled or timeout
            signaled = event.wait(timeout=remaining)
            if not signaled:
                # event.wait returned False -> timeout
                raise TimeoutError(
                    f"Timeout waiting for multicast data. "
                    f"The sender may not have called dma(ringview, mcast_addr).wait() "
                    f"or there may be a deadlock."
                )

            # Event signaled - examine queue under lock
            with lock:
                if len(queue) == 0:
                    # Spurious wakeup or another receiver consumed; wait again
                    event.clear()
                    continue

                src_data, remaining_receivers = queue[0]
                if len(dst) != len(src_data):
                    raise ValueError(
                        f"Destination RingView length ({len(dst)}) "
                        f"does not match multicast data length ({len(src_data)})"
                    )

                dst.store(src_data)

                # Decrement receiver count and update queue
                remaining_receivers -= 1
                if remaining_receivers == 0:
                    queue.popleft()
                    # If nothing left, clear the event so future waits block
                    if len(queue) == 0:
                        event.clear()
                else:
                    queue[0] = (src_data, remaining_receivers)

                return
