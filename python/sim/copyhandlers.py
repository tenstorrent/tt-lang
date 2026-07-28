# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy transfer handlers using a registry-based strategy pattern.

Each handler implements validate() and transfer() for a specific (src_type, dst_type) pair.
New transfer types can be added by creating a new handler and decorating it with
@register_copy_handler.
"""

import math
from collections import deque
from functools import cache
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
from .context_types import PipeEntry, PipeMessage
from .dfb import Block
from .pipe import (
    AnySrcPipeIdentity,
    AnyDst,
    AnyPipe,
    DstPipeIdentity,
    Pipe,
    SrcPipeIdentity,
)
from .trace import TRACE, get_pipe_name, trace
from .ttnnsim import (
    Tensor,
    check_count_match,
    tile_count_from_shape,
)
from .typedefs import IndexType, NodeCoord, Shape

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


def _is_dry_run() -> bool:
    return get_context().config.dry_run


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


# ---------------------------------------------------------------------------
# Cached shape/layout validators.
#
# Tensor/Block copy validation is a pure function of the two layouts and two
# shapes; the matmul-tutorial dry run hits the same handful of combinations
# roughly four million times.  Memoising on the four primitive arguments via
# functools.cache lets repeat calls reduce to a single dict lookup inside the
# decorator, with no per-handler bookkeeping.  Only successful results are
# cached (exceptions are not memoised by functools.cache), so the failure
# message is regenerated every call -- which is what we want.
# ---------------------------------------------------------------------------


@cache
def _validate_tensor_to_block_shapes(
    src_layout: IndexType,
    src_shape: Shape,
    dst_layout: IndexType,
    dst_shape: Shape,
) -> None:
    if src_layout != dst_layout:
        raise ValueError(
            f"Layout mismatch in Tensor -> Block copy: "
            f"source tensor has layout {src_layout.name}, "
            f"but block has layout {dst_layout.name}"
        )
    check_count_match(
        tile_count_from_shape(src_layout, src_shape),
        math.prod(dst_shape),
        src_layout,
        f"Tensor shape {src_shape}",
        f"Block shape {dst_shape}",
    )


@cache
def _validate_block_to_tensor_shapes(
    src_layout: IndexType,
    src_shape: Shape,
    dst_layout: IndexType,
    dst_shape: Shape,
) -> None:
    if src_layout != dst_layout:
        raise ValueError(
            f"Layout mismatch in Block -> Tensor copy: "
            f"source block has layout {src_layout.name}, "
            f"but destination tensor has layout {dst_layout.name}"
        )
    check_count_match(
        math.prod(src_shape),
        tile_count_from_shape(dst_layout, dst_shape),
        src_layout,
        f"Block shape {src_shape}",
        f"Tensor shape {dst_shape}",
    )


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
        """Pipe send: store data in shared buffer accessible by all nodes.

        The queued ``PipeMessage`` always records the sent block's tile-grid
        shape so the destination shape check runs identically in both modes. In
        dry-run mode the message's ``data`` is left ``None`` (no payload bytes),
        but the queue bookkeeping (receiver count, message id, receiver set) is
        still maintained so pipe sequencing and backpressure are exercised.
        """
        message = PipeMessage(
            grid_shape=src.shape,
            data=None if _is_dry_run() else src.raw_tensor,
        )

        # Get or create pipe entry atomically
        entry = _get_or_create_pipe_entry(dst)
        # Calculate number of receivers based on dst_node_range type
        num_receivers: int = 1

        # dst_node_range can be either NodeCoord or NodeRange
        dst_node_range: AnyDst = dst.dst

        # Helper predicate for pattern matching
        def has_slices(t: Any) -> bool:
            """Check if tuple contains any slice objects."""
            return len(t) > 0 and any(type(item) is slice for item in t)

        # Match on the structure of dst_node_range
        match dst_node_range:
            case int():
                # Single 1D node
                num_receivers = 1
            case tuple() if has_slices(dst_node_range):
                # NodeRange with slices: expand and count
                from .pipe import expand_node_range

                expanded_nodes: List[NodeCoord] = expand_node_range(dst_node_range)
                num_receivers = len(expanded_nodes)
            case tuple():
                # Single multi-dimensional node
                num_receivers = 1

        # Add to the queue with receiver count, message ID, and empty receiver set.
        msg_id = entry["next_msg_id"]
        entry["next_msg_id"] += 1
        entry["queue"].append((message, num_receivers, msg_id, set[int]()))

        if TRACE.enabled:
            trace(
                "pipe_send",
                pipe=get_pipe_name(dst),
                tiles=math.prod(message.grid_shape),
            )

    def can_wait(self, src: Block, dst: AnyPipe) -> bool:
        """Block to Pipe copy completes immediately on wait()."""
        return True


@register_copy_handler(Tensor, Block)
class TensorToBlockHandler:
    """Handler for TTNN.Tensor -> Block transfers using tile-level indexing."""

    def validate(self, src: Tensor, dst: Block) -> None:
        _validate_tensor_to_block_shapes(src.layout, src.shape, dst.layout, dst.shape)

    def transfer(self, src: Tensor, dst: Block) -> None:
        """Transfer tensor data into Block."""
        if _is_dry_run():
            return
        dst.copy_as_dest(src)

    def can_wait(self, src: Tensor, dst: Block) -> bool:
        return True


@register_copy_handler(Block, Tensor)
class BlockToTensorHandler:
    """Handler for Block -> TTNN.Tensor transfers using tile-level indexing."""

    def validate(self, src: Block, dst: Tensor) -> None:
        _validate_block_to_tensor_shapes(src.layout, src.shape, dst.layout, dst.shape)

    def transfer(self, src: Block, dst: Tensor) -> None:
        """Transfer Block data into tensor."""
        if _is_dry_run():
            return
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
        """Pipe to Block copy can only proceed when pipe has data for this node.

        Returns True only when there is at least one queued message that the
        current node has not yet received.  The greenlet scheduler polls this
        before calling transfer(), so transfer() can assume data is available.
        """
        pipe_buffer = get_context().copy_state.pipe_buffer
        entry = pipe_buffer.get(src)
        if entry is None or len(entry["queue"]) == 0:
            return False

        # Check whether there is a message this node has not yet received.
        try:
            from .nodecontext import node

            node_id = node(dims=1)
            return any(node_id not in recv_set for _, _, _, recv_set in entry["queue"])
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

        # Determine current node ID for per-node message tracking.
        try:
            from .nodecontext import node

            node_id = node(dims=1)
            node_id_available = True
        except (ImportError, RuntimeError):
            node_id_available = False
            node_id = None

        # Find the first message this node has not yet received.
        for idx, (message, remaining_recv, msg_id, recv_set) in enumerate(queue):
            if not node_id_available or node_id not in recv_set:
                if message.grid_shape != dst.shape:
                    raise ValueError(
                        f"Destination Block shape {dst.shape} "
                        f"does not match pipe data shape {message.grid_shape}"
                    )

                # Payload copy only happens when data is present; in dry-run the
                # message carries no bytes (data is None) and the copy is skipped
                # while the queue bookkeeping below still runs.
                if message.data is not None:
                    dst.copy_as_dest(message.data)

                if TRACE.enabled:
                    trace(
                        "pipe_recv",
                        pipe=get_pipe_name(src),
                        tiles=math.prod(message.grid_shape),
                    )

                if node_id_available:
                    match node_id:
                        case int():
                            recv_set.add(node_id)
                        case _:
                            raise TypeError("node_id should be int when dims=1")

                remaining_recv -= 1
                if remaining_recv == 0:
                    del queue[idx]
                else:
                    queue[idx] = (message, remaining_recv, msg_id, recv_set)
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
