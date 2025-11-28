# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DMA transfer handlers using a registry-based strategy pattern.

Each handler implements validate() and transfer() for a specific (src_type, dst_type) pair.
New transfer types can be added by creating a new handler and decorating it with
@register_dma_handler.
"""

from typing import Any, Dict, Protocol, Tuple, Type, Deque, List, Union
from collections import deque
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


# Global multicast buffer for simulating NoC multicast communication
# In a real implementation, this would be handled by the NoC hardware
# Each queue entry is (data, remaining_receiver_count)
# Multiple receivers can consume the same data, tracked by the count
_multicast_buffer: Dict[MulticastAddress, Deque[Tuple[List[torch.Tensor], Count]]] = {}


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

        # Initialize queue if it doesn't exist
        if dst not in _multicast_buffer:
            _multicast_buffer[dst] = deque()

        # Calculate number of receivers (all cores except the sender)
        num_receivers = len(dst.core_indices) - 1

        # Add to the queue for this multicast address with receiver count
        # In a real implementation, this would send packets over the NoC
        _multicast_buffer[dst].append((src_data, num_receivers))


@register_dma_handler(MulticastAddress, RingView)
class MulticastToRingViewHandler:
    """Handler for MulticastAddress → RingView (multicast receive)."""

    def validate(self, src: MulticastAddress, dst: RingView[torch.Tensor]) -> None:
        """Validate multicast receive - validation happens during transfer when data is available."""
        pass

    def transfer(self, src: MulticastAddress, dst: RingView[torch.Tensor]) -> None:
        """Multicast receive: retrieve data from shared multicast buffer."""
        # Poll until data is available in the multicast buffer queue
        # In a real implementation, this would be hardware-level blocking
        start_time = time.time()
        while src not in _multicast_buffer or len(_multicast_buffer[src]) == 0:
            if time.time() - start_time > DMA_MULTICAST_TIMEOUT:
                raise TimeoutError(
                    f"Timeout waiting for multicast data. "
                    f"The sender may not have called dma(ringview, mcast_addr).wait() "
                    f"or there may be a deadlock."
                )
            time.sleep(0.001)  # Small sleep to avoid busy-waiting

        # Peek at the front of the queue (don't remove yet)
        src_data, remaining_receivers = _multicast_buffer[src][0]
        if len(dst) != len(src_data):
            raise ValueError(
                f"Destination RingView length ({len(dst)}) "
                f"does not match multicast data length ({len(src_data)})"
            )
        dst.store(src_data)

        # Decrement receiver count
        remaining_receivers -= 1
        if remaining_receivers == 0:
            # All receivers consumed, remove from queue
            _multicast_buffer[src].popleft()
        else:
            # Update the count for remaining receivers
            _multicast_buffer[src][0] = (
                src_data,
                remaining_receivers,
            )
