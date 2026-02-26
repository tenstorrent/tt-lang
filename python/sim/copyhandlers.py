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

from .dfb import Block
from .constants import COPY_PIPE_TIMEOUT
from .pipe import AnySrcPipeIdentity, DstPipeIdentity, SrcPipeIdentity
from .stats import (
    record_tensor_read,
    record_tensor_write,
    record_pipe_read,
    record_pipe_write,
)
from .ttnnsim import Tensor
from .pipe import AnyDst, AnyPipe, Pipe
from .typedefs import CoreCoord, Count, Shape

if TYPE_CHECKING:
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
    AnyPipe,
    AnySrcPipeIdentity,
    DstPipeIdentity,
]
CopyEndpointType = Union[
    Type[Tensor],
    Type[Block],
    Type[AnyPipe],
    Type[AnySrcPipeIdentity],
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


# Global pipe state for simulating NoC pipe communication
# For each pipe we keep a small structure with:
# - queue: deque of (data, remaining_receiver_count, message_id, receivers_set)
# - event: threading.Event set when queue is non-empty
# - lock: threading.Lock to guard queue and receiver count updates
# - next_msg_id: counter for assigning unique IDs to messages
# In a real implementation this would be handled by NoC hardware.
class _PipeEntry(TypedDict):
    queue: Deque[
        Tuple[Tensor, Count, int, set[int]]
    ]  # (data, remaining, msg_id, receivers_who_got_it)
    event: threading.Event
    lock: threading.Lock
    next_msg_id: int


_pipe_buffer: Dict[AnyPipe, _PipeEntry] = {}
# Lock protecting creation of per-pipe entries in _pipe_buffer.
# This ensures all threads agree on the same entry object (and its lock)
# and avoids races where two threads create different entry dicts for
# the same pipe.
_pipe_registry_lock = threading.Lock()


def _get_or_create_pipe_entry(pipe: AnyPipe) -> _PipeEntry:
    """Get or create pipe buffer entry for a given pipe.

    This helper ensures atomic initialization of pipe entries so all threads
    see the same entry object and its lock.

    Args:
        pipe: The pipe to get or create an entry for

    Returns:
        The pipe entry containing queue, event, lock, and next_msg_id
    """
    with _pipe_registry_lock:
        entry = _pipe_buffer.get(pipe)
        if entry is None:
            new_entry: _PipeEntry = {
                "queue": deque(),
                "event": threading.Event(),
                "lock": threading.Lock(),
                "next_msg_id": 0,
            }
            _pipe_buffer[pipe] = new_entry
            entry = new_entry
        return entry


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

    def validate(self, src: Block, dst: AnyPipe) -> None:
        """Validate pipe send - no specific validation needed."""
        pass

    def transfer(self, src: Block, dst: AnyPipe) -> None:
        """Pipe send: store data in shared buffer accessible by all cores."""
        src_data = src.raw_tensor

        # Record pipe write statistics
        record_pipe_write(dst, src_data)

        # Get or create pipe entry atomically
        entry = _get_or_create_pipe_entry(dst)

        # Calculate number of receivers based on dst_core_range type
        num_receivers: int = 1

        # dst_core_range can be either CoreCoord or CoreRange
        dst_core_range: AnyDst = dst.dst_core_range

        # Helper predicate for pattern matching
        def has_slices(t: Any) -> bool:
            """Check if tuple contains any slice objects."""
            return len(t) > 0 and any(type(item) is slice for item in t)

        # Match on the structure of dst_core_range
        match dst_core_range:
            case int():
                # Single 1D core
                num_receivers = 1
            case tuple() if has_slices(dst_core_range):
                # CoreRange with slices: expand and count
                from .pipe import expand_core_range

                expanded_cores: List[CoreCoord] = expand_core_range(dst_core_range)
                num_receivers = len(expanded_cores)
            case tuple():
                # Single multi-dimensional core
                num_receivers = 1

        # Add to the queue for this pipe with receiver count, message ID, and empty receiver set
        # and notify any waiting receivers via event
        with entry["lock"]:
            msg_id = entry["next_msg_id"]
            entry["next_msg_id"] += 1
            entry["queue"].append((src_data, num_receivers, msg_id, set[int]()))
            # Signal that data is available
            entry["event"].set()

    def can_wait(self, src: Block, dst: AnyPipe) -> bool:
        """Block to Pipe copy completes immediately on wait()."""
        return True


@register_copy_handler(Tensor, Block)
class TensorToBlockHandler:
    """Handler for TTNN.Tensor -> Block transfers using tile-level indexing."""

    def validate(self, src: Tensor, dst: Block) -> None:
        from .dfb import tile_count_from_tensor
        import math

        src_tiles = tile_count_from_tensor(src)
        dst_tiles = math.prod(dst.shape)
        if src_tiles != dst_tiles:
            raise ValueError(
                f"Tensor shape {src.shape} does not match Block shape {dst.shape} "
                f"(tile counts: {src_tiles} vs {dst_tiles})"
            )

    def transfer(self, src: Tensor, dst: Block) -> None:
        """Transfer tensor data into Block."""
        record_tensor_read(src)
        dst.copy_as_dest(src)

    def can_wait(self, src: Tensor, dst: Block) -> bool:
        return True


@register_copy_handler(Block, Tensor)
class BlockToTensorHandler:
    """Handler for Block -> TTNN.Tensor transfers using tile-level indexing."""

    def validate(self, src: Block, dst: Tensor) -> None:
        from .dfb import tile_count_from_tensor
        import math

        src_tiles = math.prod(src.shape)
        dst_tiles = tile_count_from_tensor(dst)
        if src_tiles != dst_tiles:
            raise ValueError(
                f"Block shape {src.shape} does not match Tensor shape {dst.shape} "
                f"(tile counts: {src_tiles} vs {dst_tiles})"
            )

    def transfer(self, src: Block, dst: Tensor) -> None:
        """Transfer Block data into tensor."""
        record_tensor_write(dst)
        dst_raw = dst.to_torch()
        src_raw = src.raw_tensor.to_torch()
        dst_raw.copy_(src_raw.reshape(dst_raw.shape))

    def can_wait(self, src: Block, dst: Tensor) -> bool:
        return True


@register_copy_handler(Pipe, Block)
class PipeToBlockHandler:
    """Handler for Pipe → Block (pipe receive)."""

    def validate(self, src: AnyPipe, dst: Block) -> None:
        """Validate pipe receive - validation happens during transfer when data is available."""
        pass

    def can_wait(self, src: AnyPipe, dst: Block) -> bool:
        """Pipe to Block copy can only proceed when pipe has data."""
        # Check if pipe has data available without blocking
        with _pipe_registry_lock:
            entry = _pipe_buffer.get(src)
            if entry is None:
                return False

        with entry["lock"]:
            return len(entry["queue"]) > 0

    def transfer(self, src: AnyPipe, dst: Block) -> None:
        """Pipe receive: retrieve data from shared pipe buffer."""
        # Use an event to wait for data instead of polling. This reduces CPU
        # usage and provides a cleaner synchronization primitive for tests.
        start_time = time.time()

        # Get or create pipe entry atomically
        entry = _get_or_create_pipe_entry(src)
        event: threading.Event = entry["event"]
        queue: Deque[Tuple[Tensor, Count, int, set[int]]] = entry["queue"]
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

                # Get current core ID for tracking which messages this core has received
                try:
                    from .corecontext import core

                    core_id = core(dims=1)
                    core_id_available = True
                except (ImportError, RuntimeError):
                    # Non-kernel context or core not available - no tracking needed
                    core_id_available = False
                    core_id = None

                # Find the first message in the queue that this core hasn't received yet
                # This handles buffer_factor>1 where the same core may have multiple
                # pending receives but should get different messages
                src_data: Tensor | None = None
                remaining_receivers: Count | None = None
                msg_id_to_recv: int | None = None
                receivers_set: set[int] | None = None
                msg_index = 0

                for idx, (msg_data, remaining_recv, msg_id, recv_set) in enumerate(
                    queue
                ):
                    if not core_id_available or core_id not in recv_set:
                        # This core hasn't received this message yet
                        src_data = msg_data
                        remaining_receivers = remaining_recv
                        msg_id_to_recv = msg_id
                        receivers_set = recv_set
                        msg_index = idx
                        break

                if src_data is None:
                    # All messages in queue have already been received by this core
                    # Wait for new messages to arrive
                    event.clear()
                    continue

                # At this point, all variables are guaranteed to be non-None
                assert remaining_receivers is not None
                assert msg_id_to_recv is not None
                assert receivers_set is not None

                # Mark this core as having received the message
                if core_id_available:
                    match core_id:
                        case int():
                            receivers_set.add(core_id)
                        case _:
                            raise TypeError("core_id should be int when dims=1")

                if src_data.shape != dst.raw_tensor.shape:
                    raise ValueError(
                        f"Destination Block shape {dst.raw_tensor.shape} "
                        f"does not match pipe data shape {src_data.shape}"
                    )

                dst.copy_as_dest(src_data)

                # Record pipe read statistics
                record_pipe_read(src, src_data)

                # Decrement receiver count and update queue
                remaining_receivers -= 1

                if remaining_receivers == 0:
                    # All receivers got this message, remove it from queue
                    del queue[msg_index]
                    # If nothing left, clear the event so future waits block
                    if len(queue) == 0:
                        event.clear()
                else:
                    # Update the message in place with new remaining count and receiver set
                    queue[msg_index] = (
                        src_data,
                        remaining_receivers,
                        msg_id_to_recv,
                        receivers_set,
                    )

                return


# ===== Pipe Identity Wrapper Handlers =====
# These handlers delegate to the underlying Pipe handlers for SrcPipeIdentity and DstPipeIdentity


@register_copy_handler(Block, SrcPipeIdentity)
class BlockToSrcPipeIdentityHandler:
    """Handler for Block → SrcPipeIdentity (delegates to Block → Pipe)."""

    def __init__(self) -> None:
        self._delegate: CopyTransferHandler | None = None

    def _get_delegate(self) -> CopyTransferHandler:
        """Lazy initialization of delegate handler."""
        if self._delegate is None:
            self._delegate = handler_registry[(Block, Pipe)]
        return self._delegate

    def validate(self, src: Block, dst: AnySrcPipeIdentity) -> None:
        # Delegate to the Pipe handler
        self._get_delegate().validate(src, dst.pipe)

    def transfer(self, src: Block, dst: AnySrcPipeIdentity) -> None:
        # Delegate to the Pipe handler
        self._get_delegate().transfer(src, dst.pipe)

    def can_wait(self, src: Block, dst: AnySrcPipeIdentity) -> bool:
        return self._get_delegate().can_wait(src, dst.pipe)


@register_copy_handler(DstPipeIdentity, Block)
class DstPipeIdentityToBlockHandler:
    """Handler for DstPipeIdentity → Block (delegates to Pipe → Block)."""

    def __init__(self) -> None:
        self._delegate: CopyTransferHandler | None = None

    def _get_delegate(self) -> CopyTransferHandler:
        """Lazy initialization of delegate handler."""
        if self._delegate is None:
            self._delegate = handler_registry[(Pipe, Block)]
        return self._delegate

    def validate(self, src: DstPipeIdentity, dst: Block) -> None:
        # Delegate to the Pipe handler
        self._get_delegate().validate(src.pipe, dst)

    def transfer(self, src: DstPipeIdentity, dst: Block) -> None:
        # Delegate to the Pipe handler
        self._get_delegate().transfer(src.pipe, dst)

    def can_wait(self, src: DstPipeIdentity, dst: Block) -> bool:
        return self._get_delegate().can_wait(src.pipe, dst)
