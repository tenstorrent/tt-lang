# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy transfer handlers using a registry-based strategy pattern.

Each handler implements validate() and transfer() for a specific (src_type, dst_type) pair.
New transfer types can be added by creating a new handler and decorating it with
@register_copy_handler.
"""

from typing import Any, Dict, Protocol, Tuple, Type, Deque, List, Union, TypedDict
from collections import deque
import threading
import time
import torch
from . import torch_utils as tu
from .block import Block
from .constants import TILE_SHAPE, COPY_PIPE_TIMEOUT
from .typedefs import Pipe, Count


# TODO: Ideally, to avoid duplication, we would want something like this:
# CopyEndpointTypes: List[type] = [torch.Tensor, Block[torch.Tensor], Pipe]
# CopyEndpoint = Union[*CopyEndpointTypes]
# CopyEndpointType = Union[*[Type[x] for x in CopyEndpointTypes]]
#
# Unfortunately, this is too difficult for static analysis to understand
# (pyright, it needs to execute the expansion to figure it out). So we stick to
# the simpler explicit definition bellow.

# Copy endpoint types - these are the valid types for copy transfers
# To add a new endpoint type, add it to the Unions and implement a handler for it
CopyEndpoint = Union[torch.Tensor, Block[torch.Tensor], Pipe]
CopyEndpointType = Union[Type[torch.Tensor], Type[Block[torch.Tensor]], Type[Pipe]]


# Global pipe state for simulating NoC pipe communication
# For each pipe we keep a small structure with:
# - queue: deque of (data, remaining_receiver_count)
# - event: threading.Event set when queue is non-empty
# - lock: threading.Lock to guard queue and receiver count updates
# In a real implementation this would be handled by NoC hardware.
class _PipeEntry(TypedDict):
    queue: Deque[Tuple[List[torch.Tensor], Count]]
    event: threading.Event
    lock: threading.Lock


_pipe_buffer: Dict[Pipe, _PipeEntry] = {}
# Lock protecting creation of per-pipe entries in _pipe_buffer.
# This ensures all threads agree on the same entry object (and its lock)
# and avoids races where two threads create different entry dicts for
# the same pipe.
_pipe_registry_lock = threading.Lock()


class CopyTransferHandler(Protocol):
    """Protocol for copy transfer handlers."""

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
handler_registry: Dict[
    Tuple[CopyEndpointType, CopyEndpointType], CopyTransferHandler
] = {}


def register_copy_handler(src_type: CopyEndpointType, dst_type: CopyEndpointType):
    """
    Decorator to register a copy transfer handler for a specific (src_type, dst_type) pair.

    Args:
        src_type: Source type class (must be a valid copy endpoint type)
        dst_type: Destination type class (must be a valid copy endpoint type)

    Returns:
        Decorator function

    Example:
        @register_copy_handler(torch.Tensor, Block)
        class TensorToBlockHandler:
            def validate(self, src, dst): ...
            def transfer(self, src, dst): ...
    """

    def decorator(handler_cls: Type[CopyTransferHandler]):
        handler_registry[(src_type, dst_type)] = handler_cls()
        return handler_cls

    return decorator


@register_copy_handler(torch.Tensor, Block)
class TensorToBlockHandler:
    """Handler for Tensor → Block transfers."""

    def validate(self, src: torch.Tensor, dst: Block[torch.Tensor]) -> None:
        """Validate tensor to Block transfer."""
        if len(src.shape) != 2:
            raise ValueError(
                f"Copy only supports 2D tensors, got {len(src.shape)}D tensor with shape {src.shape}"
            )

        num_tiles = tu.tile_count(src.shape, TILE_SHAPE)
        expected_tiles = len(dst)

        if num_tiles != expected_tiles:
            raise ValueError(
                f"Tensor contains {num_tiles} tiles but Block has {expected_tiles} slots"
            )

    def transfer(self, src: torch.Tensor, dst: Block[torch.Tensor]) -> None:
        """Transfer tensor data to Block by splitting into tiles."""
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


@register_copy_handler(Block, torch.Tensor)
class BlockToTensorHandler:
    """Handler for Block → Tensor transfers."""

    def validate(self, src: Block[torch.Tensor], dst: torch.Tensor) -> None:
        """Validate Block to tensor transfer."""
        if len(dst.shape) != 2:
            raise ValueError(
                f"Copy only supports 2D tensors, got {len(dst.shape)}D tensor with shape {dst.shape}"
            )

        dst_tiles = tu.tile_count(dst.shape, TILE_SHAPE)
        if len(src) != dst_tiles:
            raise ValueError(f"Expected {len(src)} tiles but found {dst_tiles}")

    def transfer(self, src: Block[torch.Tensor], dst: torch.Tensor) -> None:
        """Transfer Block data to tensor by combining tiles."""
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


@register_copy_handler(Block, Pipe)
class BlockToPipeHandler:
    """Handler for Block → Pipe (pipe send)."""

    def validate(self, src: Block[torch.Tensor], dst: Pipe) -> None:
        """Validate pipe send - no specific validation needed."""
        pass

    def transfer(self, src: Block[torch.Tensor], dst: Pipe) -> None:
        """Pipe send: store data in shared buffer accessible by all cores."""
        src_data = [src[i] for i in range(len(src))]
        # Initialize per-pipe state atomically so all threads see the
        # same entry (and therefore the same per-entry lock).
        with _pipe_registry_lock:
            entry = _pipe_buffer.get(dst)
            if entry is None:
                new_entry: _PipeEntry = {
                    "queue": deque(),
                    "event": threading.Event(),
                    "lock": threading.Lock(),
                }
                _pipe_buffer[dst] = new_entry
                entry = new_entry

        # Calculate number of receivers based on dst_core_range type
        match dst.dst_core_range:
            case (tuple() as first, tuple() as second):
                # Rectangular range: count all cores in the rectangle
                dims = len(first)
                num_receivers = 1
                for i in range(dims):
                    range_size = abs(second[i] - first[i]) + 1
                    num_receivers *= range_size
            case tuple():
                # Single multi-dimensional core
                num_receivers = 1
            case int():
                # Single 1D core
                num_receivers = 1

        # Add to the queue for this pipe with receiver count
        # and notify any waiting receivers via event
        with entry["lock"]:
            entry["queue"].append((src_data, num_receivers))
            # Signal that data is available
            entry["event"].set()


@register_copy_handler(Pipe, Block)
class PipeToBlockHandler:
    """Handler for Pipe → Block (pipe receive)."""

    def validate(self, src: Pipe, dst: Block[torch.Tensor]) -> None:
        """Validate pipe receive - validation happens during transfer when data is available."""
        pass

    def transfer(self, src: Pipe, dst: Block[torch.Tensor]) -> None:
        """Pipe receive: retrieve data from shared pipe buffer."""
        # Use an event to wait for data instead of polling. This reduces CPU
        # usage and provides a cleaner synchronization primitive for tests.
        start_time = time.time()

        # Ensure entry exists atomically so we can safely access event/lock.
        with _pipe_registry_lock:
            entry = _pipe_buffer.get(src)
            if entry is None:
                new_entry: _PipeEntry = {
                    "queue": deque(),
                    "event": threading.Event(),
                    "lock": threading.Lock(),
                }
                _pipe_buffer[src] = new_entry
                entry = new_entry
        event: threading.Event = entry["event"]
        queue: Deque[Tuple[List[torch.Tensor], Count]] = entry["queue"]
        lock: threading.Lock = entry["lock"]

        while True:
            # Compute remaining timeout
            elapsed = time.time() - start_time
            remaining = COPY_PIPE_TIMEOUT - elapsed
            if remaining <= 0:
                raise TimeoutError(
                    f"Timeout waiting for pipe data. "
                    f"The sender may not have called copy(block, pipe).wait() "
                    f"or there may be a deadlock."
                )

            # Wait until signaled or timeout
            signaled = event.wait(timeout=remaining)
            if not signaled:
                # event.wait returned False -> timeout
                raise TimeoutError(
                    f"Timeout waiting for pipe data. "
                    f"The sender may not have called copy(block, pipe).wait() "
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
                        f"Destination Block length ({len(dst)}) "
                        f"does not match pipe data length ({len(src_data)})"
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
