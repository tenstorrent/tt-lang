# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy transfer handlers using a registry-based strategy pattern.

Each handler implements validate() and transfer() for a specific (src_type, dst_type) pair.
New transfer types can be added by creating a new handler and decorating it with
@register_copy_handler.
"""

from collections import deque
from typing import (
    Any,
    Dict,
    Final,
    List,
    Protocol,
    Tuple,
    Type,
    Union,
)

from .context import get_context
from .context_types import PipeEntry
from .dfb import Block
from .pipe import AnySrcPipeIdentity, DstPipeIdentity, SrcPipeIdentity
from .stats import (
    record_tensor_read,
    record_tensor_write,
    record_pipe_read,
    record_pipe_write,
)
from .ttnnsim import Tensor
from .pipe import AnyDst, AnyPipe, Pipe
from .typedefs import CoreCoord
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


def _get_or_create_pipe_entry(pipe: AnyPipe) -> PipeEntry:
    """Get or create the pipe buffer entry for a given pipe."""
    pipe_buffer = get_context().copy_state.pipe_buffer
    entry = pipe_buffer.get(pipe)
    if entry is None:
        new_entry: PipeEntry = {"queue": deque(), "next_msg_id": 0}
        pipe_buffer[pipe] = new_entry
        return new_entry
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


# Handler registry: (src_type, dst_type) -> handler instance
# Static lookup table populated at import time via @register_copy_handler decorators.
# Uses uppercase naming and Final to indicate this is a constant that should not be reassigned.
HANDLER_REGISTRY: Final[
    Dict[Tuple[CopyEndpointType, CopyEndpointType], CopyTransferHandler]
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
        # Register handler in module-level registry
        HANDLER_REGISTRY[(src_type, dst_type)] = handler_cls()
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

        # Add to the queue with receiver count, message ID, and empty receiver set.
        msg_id = entry["next_msg_id"]
        entry["next_msg_id"] += 1
        entry["queue"].append((src_data, num_receivers, msg_id, set[int]()))

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
        """Pipe to Block copy can only proceed when pipe has data for this core.

        Returns True only when there is at least one queued message that the
        current core has not yet received.  The greenlet scheduler polls this
        before calling transfer(), so transfer() can assume data is available.
        """
        pipe_buffer = get_context().copy_state.pipe_buffer
        entry = pipe_buffer.get(src)
        if entry is None or len(entry["queue"]) == 0:
            return False

        # Check whether there is a message this core has not yet received.
        try:
            from .corecontext import core

            core_id = core(dims=1)
            return any(core_id not in recv_set for _, _, _, recv_set in entry["queue"])
        except (ImportError, RuntimeError):
            # Non-kernel context: any queued message is receivable.
            return True

    def transfer(self, src: AnyPipe, dst: Block) -> None:
        """Pipe receive: dequeue one message from the pipe buffer.

        The greenlet scheduler guarantees can_wait() returned True immediately
        before this call, so a receivable message is always present.
        """
        entry = _get_or_create_pipe_entry(src)
        queue = entry["queue"]

        # Determine current core ID for per-core message tracking.
        try:
            from .corecontext import core

            core_id = core(dims=1)
            core_id_available = True
        except (ImportError, RuntimeError):
            core_id_available = False
            core_id = None

        # Find the first message this core has not yet received.
        for idx, (msg_data, remaining_recv, msg_id, recv_set) in enumerate(queue):
            if not core_id_available or core_id not in recv_set:
                if msg_data.shape != dst.raw_tensor.shape:
                    raise ValueError(
                        f"Destination Block shape {dst.raw_tensor.shape} "
                        f"does not match pipe data shape {msg_data.shape}"
                    )

                dst.copy_as_dest(msg_data)
                record_pipe_read(src, msg_data)

                if core_id_available:
                    match core_id:
                        case int():
                            recv_set.add(core_id)
                        case _:
                            raise TypeError("core_id should be int when dims=1")

                remaining_recv -= 1
                if remaining_recv == 0:
                    del queue[idx]
                else:
                    queue[idx] = (msg_data, remaining_recv, msg_id, recv_set)
                return

        # Unreachable if can_wait() was accurate.
        raise RuntimeError("transfer() called but no receivable message in pipe queue")


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
            self._delegate = HANDLER_REGISTRY[(Block, Pipe)]
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
            self._delegate = HANDLER_REGISTRY[(Pipe, Block)]
        return self._delegate

    def validate(self, src: DstPipeIdentity, dst: Block) -> None:
        # Delegate to the Pipe handler
        self._get_delegate().validate(src.pipe, dst)

    def transfer(self, src: DstPipeIdentity, dst: Block) -> None:
        # Delegate to the Pipe handler
        self._get_delegate().transfer(src.pipe, dst)

    def can_wait(self, src: DstPipeIdentity, dst: Block) -> bool:
        return self._get_delegate().can_wait(src.pipe, dst)
