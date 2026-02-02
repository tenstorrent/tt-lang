# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy transfer handlers using a registry-based strategy pattern.

Each handler implements validate() and transfer() for a specific (src_type, dst_type) pair.
New transfer types can be added by creating a new handler and decorating it with
@register_copy_handler.
"""

import threading
import time
from collections import deque
from numpy import prod
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    List,
    Protocol,
    Tuple,
    Type,
    TypedDict,
    Union,
)

from .block import Block
from .cb import ReserveContext, WaitContext
from .constants import COPY_PIPE_TIMEOUT, TILE_SHAPE
from .pipe import DstPipeIdentity, SrcPipeIdentity
from .ttnnsim import Tensor, tensor_shape_in_tiles
from .typedefs import Count, Pipe, Shape

if TYPE_CHECKING:
    from .cb import ReserveContext, WaitContext
    from .pipe import SrcPipeIdentity


# TODO: Ideally, to avoid duplication, we would want something like this:
# CopyEndpointTypes: List[type] = [torch.Tensor, Block, Pipe]
# CopyEndpoint = Union[*CopyEndpointTypes]
# CopyEndpointType = Union[*[Type[x] for x in CopyEndpointTypes]]
#
# Unfortunately, this is too difficult for static analysis to understand
# (pyright, it needs to execute the expansion to figure it out). So we stick to
# the simpler explicit definition bellow.

# Copy endpoint types - these are the valid types for copy transfers
# To add a new endpoint type, add it to the Unions and implement a handler for it
CopyEndpoint = Union[
    Tensor,
    Block,
    Pipe,
    "ReserveContext",
    "WaitContext",
    "SrcPipeIdentity",
    DstPipeIdentity,
]
CopyEndpointType = Union[
    Type[Tensor],
    Type[Block],
    Type[Pipe],
    Type["ReserveContext"],
    Type["WaitContext"],
    Type["SrcPipeIdentity"],
    Type[DstPipeIdentity],
]


# Tile calculation utilities
def tile_count(tensor_shape: Shape, tile_shape: Shape) -> Count:
    """
    Calculate the total number of tiles in a tensor.

    Args:
        tensor_shape: Shape of the tensor (height, width, ...)
        tile_shape: Shape of each tile (height, width, ...)

    Returns:
        Total number of tiles needed to represent the tensor

    Example:
        For a (64, 128) tensor with tile_shape=(32, 32):
        tile_count((64, 128), (32, 32)) = (64//32) * (128//32) = 2 * 4 = 8 tiles
    """

    if len(tensor_shape) != len(tile_shape):
        raise ValueError(
            f"tensor_shape and tile_shape must have same dimensions: {len(tensor_shape)} vs {len(tile_shape)}"
        )
    return int(
        prod(
            [
                tensor_dim // tile_dim
                for tensor_dim, tile_dim in zip(tensor_shape, tile_shape)
            ]
        )
    )


def tensor_shape_in_tiles_with_skip(
    tensor_shape: Shape, tile_shape: Shape
) -> Tuple[int, ...]:
    """Convert tensor shape to tile dimensions, preserving size-1 dimensions.

    Unlike tensor_shape_in_tiles, this returns 1 for dimensions that are already
    size 1, rather than attempting to divide by tile dimension. This allows
    tensors like (N, 1) or (1, N) to be properly validated against blocks.

    Args:
        tensor_shape: Shape of the tensor (height, width, ...)
        tile_shape: Shape of each tile (height, width, ...)

    Returns:
        Shape in tiles, with size-1 dimensions preserved as 1

    Example:
        tensor_shape_in_tiles_with_skip((2048, 1), (32, 32)) = (64, 1)
        tensor_shape_in_tiles_with_skip((1, 64), (32, 32)) = (1, 2)
    """
    if len(tensor_shape) != len(tile_shape):
        raise ValueError(
            f"tensor_shape and tile_shape must have same dimensions: "
            f"{len(tensor_shape)} vs {len(tile_shape)}"
        )
    return tuple(
        1 if dim_size == 1 else dim_size // tile_dim
        for dim_size, tile_dim in zip(tensor_shape, tile_shape)
    )


# Global pipe state for simulating NoC pipe communication
# For each pipe we keep a small structure with:
# - queue: deque of (data, remaining_receiver_count)
# - event: threading.Event set when queue is non-empty
# - lock: threading.Lock to guard queue and receiver count updates
# In a real implementation this would be handled by NoC hardware.
class _PipeEntry(TypedDict):
    queue: Deque[Tuple[List[Tensor], Count]]
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

    def can_wait(self, src: Any, dst: Any) -> bool:
        """
        Check if wait() can proceed without blocking.

        Args:
            src: Source object
            dst: Destination object

        Returns:
            True if the transfer can complete without blocking
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
        @register_copy_handler(Tensor, Block)
        class TensorToBlockHandler:
            def validate(self, src, dst): ...
            def transfer(self, src, dst): ...
    """

    def decorator(handler_cls: Type[CopyTransferHandler]):
        handler_registry[(src_type, dst_type)] = handler_cls()
        return handler_cls

    return decorator


@register_copy_handler(Block, Pipe)
class BlockToPipeHandler:
    """Handler for Block → Pipe (pipe send)."""

    def validate(self, src: Block, dst: Pipe) -> None:
        """Validate pipe send - no specific validation needed."""
        pass

    def transfer(self, src: Block, dst: Pipe) -> None:
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
        num_receivers = 1

        # Check if it's a CoreRange with slices (by examining elements)
        if isinstance(dst.dst_core_range, tuple) and len(dst.dst_core_range) > 0:
            has_slice = any(isinstance(item, slice) for item in dst.dst_core_range)

            if has_slice:
                # CoreRange with slices: expand and count
                from .pipe import expand_core_range
                from .typedefs import CoreRange

                expanded_cores = expand_core_range(dst.dst_core_range)  # type: ignore[arg-type]
                num_receivers = len(expanded_cores)
            elif (
                all(isinstance(item, tuple) for item in dst.dst_core_range)
                and len(dst.dst_core_range) == 2
            ):
                # Legacy rectangular range: ((x1,y1), (x2,y2))
                first, second = dst.dst_core_range
                dims = len(first)  # type: ignore[arg-type]
                for i in range(dims):
                    range_size = abs(second[i] - first[i]) + 1  # type: ignore[index]
                    num_receivers *= range_size
            else:
                # Single multi-dimensional core
                num_receivers = 1
        elif isinstance(dst.dst_core_range, int):
            # Single 1D core
            num_receivers = 1

        # Add to the queue for this pipe with receiver count
        # and notify any waiting receivers via event
        with entry["lock"]:
            entry["queue"].append((src_data, num_receivers))
            # Signal that data is available
            entry["event"].set()

    def can_wait(self, src: Block, dst: Pipe) -> bool:
        """Block to Pipe copy completes immediately on wait()."""
        return True


@register_copy_handler(Tensor, Block)
class TensorToBlockHandler:
    """Handler for TTNN.Tensor → Block transfers using tile-level indexing."""

    def validate(self, src: Tensor, dst: Block) -> None:
        if len(src.shape) != 2:
            raise ValueError(f"Tensor must be 2-dimensional, got shape {src.shape}")

        # Validate tensor shape matches block shape (in tiles)
        block_shape = dst.shape
        src_shape_in_tiles = tensor_shape_in_tiles_with_skip(src.shape, TILE_SHAPE)
        if src_shape_in_tiles != block_shape:
            raise ValueError(
                f"Tensor shape {src.shape} (={src_shape_in_tiles} tiles) does not match "
                f"Block shape {block_shape} tiles (={tuple(d * t for d, t in zip(block_shape, TILE_SHAPE))} elements)"
            )

    def transfer(self, src: Tensor, dst: Block) -> None:
        """Transfer tensor data to Block using tile-level indexing.

        Extracts tiles from src using tile coordinates and stores them as
        ttnn.Tensor objects in the Block slots.
        """
        # Calculate tile count, handling size-1 dimensions properly
        shape_in_tiles = tensor_shape_in_tiles_with_skip(src.shape, TILE_SHAPE)
        num_tiles = int(prod(shape_in_tiles))
        width_tiles = shape_in_tiles[1]

        tiles = []
        for tile_idx in range(num_tiles):
            # Convert linear index to 2D tile coordinates
            h_tile = tile_idx // width_tiles
            w_tile = tile_idx % width_tiles

            # Extract single tile using tile coordinates [h:h+1, w:w+1]
            tile = src[h_tile : h_tile + 1, w_tile : w_tile + 1]
            tiles.append(tile)

        dst.copy_as_dest(tiles)

    def can_wait(self, src: Tensor, dst: Block) -> bool:
        return True


@register_copy_handler(Block, Tensor)
class BlockToTensorHandler:
    """Handler for Block → TTNN.Tensor transfers using tile-level indexing."""

    def validate(self, src: Block, dst: Tensor) -> None:
        # Validate tensor is 2D
        if len(dst.shape) != 2:
            raise ValueError(f"Tensor must be 2-dimensional, got shape {dst.shape}")

        # Validate tensor shape matches block shape (in tiles)
        block_shape = src.shape
        dst_shape_in_tiles = tensor_shape_in_tiles_with_skip(dst.shape, TILE_SHAPE)
        if dst_shape_in_tiles != block_shape:
            raise ValueError(
                f"Tensor shape {dst.shape} (={dst_shape_in_tiles} tiles) does not match "
                f"Block shape {block_shape} tiles (={tuple(d * t for d, t in zip(block_shape, TILE_SHAPE))} elements)"
            )

    def transfer(self, src: Block, dst: Tensor) -> None:
        """Transfer Block data to tensor using tile-level indexing.

        Retrieves ttnn.Tensor objects from Block slots and places them into
        the destination tensor using tile coordinates.
        """
        # Calculate tile count, handling size-1 dimensions properly
        shape_in_tiles = tensor_shape_in_tiles_with_skip(dst.shape, TILE_SHAPE)
        dst_tiles = int(prod(shape_in_tiles))
        width_tiles = shape_in_tiles[1]

        for tile_idx in range(dst_tiles):
            # Convert linear index to 2D tile coordinates
            h_tile = tile_idx // width_tiles
            w_tile = tile_idx % width_tiles

            # Get tile from Block (this is a ttnn.Tensor)
            tile = src[tile_idx]

            # Place tile into destination using tile coordinates [h:h+1, w:w+1]
            dst[h_tile : h_tile + 1, w_tile : w_tile + 1] = tile

    def can_wait(self, src: Block, dst: Tensor) -> bool:
        return True


@register_copy_handler(Pipe, Block)
class PipeToBlockHandler:
    """Handler for Pipe → Block (pipe receive)."""

    def validate(self, src: Pipe, dst: Block) -> None:
        """Validate pipe receive - validation happens during transfer when data is available."""
        pass

    def can_wait(self, src: Pipe, dst: Block) -> bool:
        """Pipe to Block copy can only proceed when pipe has data."""
        # Check if pipe has data available without blocking
        with _pipe_registry_lock:
            entry = _pipe_buffer.get(src)
            if entry is None:
                return False

        with entry["lock"]:
            return len(entry["queue"]) > 0

    def transfer(self, src: Pipe, dst: Block) -> None:
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
        queue: Deque[Tuple[List[Tensor], Count]] = entry["queue"]
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

                dst.copy_as_dest(src_data)

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


# ===== Pipe Identity Wrapper Handlers =====
# These handlers delegate to the underlying Pipe handlers for SrcPipeIdentity and DstPipeIdentity


@register_copy_handler(Block, SrcPipeIdentity)
class BlockToSrcPipeIdentityHandler:
    """Handler for Block → SrcPipeIdentity (delegates to Block → Pipe)."""

    def validate(self, src: Block, dst: SrcPipeIdentity) -> None:
        # Delegate to the Pipe handler
        BlockToPipeHandler().validate(src, dst.pipe)

    def transfer(self, src: Block, dst: SrcPipeIdentity) -> None:
        # Delegate to the Pipe handler
        BlockToPipeHandler().transfer(src, dst.pipe)

    def can_wait(self, src: Block, dst: SrcPipeIdentity) -> bool:
        return BlockToPipeHandler().can_wait(src, dst.pipe)


@register_copy_handler(DstPipeIdentity, Block)
class DstPipeIdentityToBlockHandler:
    """Handler for DstPipeIdentity → Block (delegates to Pipe → Block)."""

    def validate(self, src: DstPipeIdentity, dst: Block) -> None:
        # Delegate to the Pipe handler
        PipeToBlockHandler().validate(src.pipe, dst)

    def transfer(self, src: DstPipeIdentity, dst: Block) -> None:
        # Delegate to the Pipe handler
        PipeToBlockHandler().transfer(src.pipe, dst)

    def can_wait(self, src: DstPipeIdentity, dst: Block) -> bool:
        return PipeToBlockHandler().can_wait(src.pipe, dst)


# ===== Context Manager Wrapper Handlers =====
# These handlers delegate to the underlying Block handlers for _ReserveContext and _WaitContext


# Tensor → ReserveContext (delegates to Tensor → Block)
@register_copy_handler(Tensor, ReserveContext)
class TensorToReserveContextHandler:
    """Handler for Tensor → ReserveContext (delegates to Tensor → Block)."""

    def validate(self, src: Tensor, dst: "ReserveContext") -> None:
        # Delegate to the Block handler
        TensorToBlockHandler().validate(src, dst.block())

    def transfer(self, src: Tensor, dst: "ReserveContext") -> None:
        # Delegate to the Block handler
        TensorToBlockHandler().transfer(src, dst.block())

    def can_wait(self, src: Tensor, dst: "ReserveContext") -> bool:
        # Delegate to the Block handler
        return TensorToBlockHandler().can_wait(src, dst.block())


# WaitContext → Tensor (delegates to Block → Tensor)
@register_copy_handler(WaitContext, Tensor)
class WaitContextToTensorHandler:
    """Handler for WaitContext → Tensor (delegates to Block → Tensor)."""

    def validate(self, src: "WaitContext", dst: Tensor) -> None:
        # Delegate to the Block handler
        BlockToTensorHandler().validate(src.block(), dst)

    def transfer(self, src: "WaitContext", dst: Tensor) -> None:
        # Delegate to the Block handler
        BlockToTensorHandler().transfer(src.block(), dst)

    def can_wait(self, src: "WaitContext", dst: Tensor) -> bool:
        # Delegate to the Block handler
        return BlockToTensorHandler().can_wait(src.block(), dst)


# WaitContext → Pipe (delegates to Block → Pipe)
@register_copy_handler(WaitContext, Pipe)
class WaitContextToPipeHandler:
    """Handler for WaitContext → Pipe (delegates to Block → Pipe)."""

    def validate(self, src: "WaitContext", dst: Pipe) -> None:
        # Delegate to the Block handler
        BlockToPipeHandler().validate(src.block(), dst)

    def transfer(self, src: "WaitContext", dst: Pipe) -> None:
        # Delegate to the Block handler
        BlockToPipeHandler().transfer(src.block(), dst)

    def can_wait(self, src: "WaitContext", dst: Pipe) -> bool:
        # Delegate to the Block handler
        return BlockToPipeHandler().can_wait(src.block(), dst)


# Pipe → ReserveContext (delegates to Pipe → Block)
@register_copy_handler(Pipe, ReserveContext)
class PipeToReserveContextHandler:
    """Handler for Pipe → ReserveContext (delegates to Pipe → Block)."""

    def validate(self, src: Pipe, dst: "ReserveContext") -> None:
        # Delegate to the Block handler
        PipeToBlockHandler().validate(src, dst.block())

    def transfer(self, src: Pipe, dst: "ReserveContext") -> None:
        # Delegate to the Block handler
        PipeToBlockHandler().transfer(src, dst.block())

    def can_wait(self, src: Pipe, dst: "ReserveContext") -> bool:
        # Delegate to the Block handler
        return PipeToBlockHandler().can_wait(src, dst.block())


# ReserveContext → Pipe (delegates to Block → Pipe)
@register_copy_handler(ReserveContext, Pipe)
class ReserveContextToPipeHandler:
    """Handler for ReserveContext → Pipe (delegates to Block → Pipe)."""

    def validate(self, src: "ReserveContext", dst: Pipe) -> None:
        # Delegate to the Block handler
        BlockToPipeHandler().validate(src.block(), dst)

    def transfer(self, src: "ReserveContext", dst: Pipe) -> None:
        # Delegate to the Block handler
        BlockToPipeHandler().transfer(src.block(), dst)

    def can_wait(self, src: "ReserveContext", dst: Pipe) -> bool:
        # Delegate to the Block handler
        return BlockToPipeHandler().can_wait(src.block(), dst)
