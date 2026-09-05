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
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import torch

from .context import get_context
from .context_types import ByteCopyFormat, PipeEntry, PipeMessage
from .dfb import Block, BlockAcquisition
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
    TILE_LAYOUT,
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

    def validate(self, src: Any, dst: Any, byte_count: Optional[int] = None) -> None:
        """
        Validate that the transfer can be performed.

        Args:
            src: Source object
            dst: Destination object
            byte_count: Optional byte count for DFB and pipe copies

        Raises:
            ValueError: If the transfer is not valid (shape mismatch, etc.)
        """
        ...

    def transfer(self, src: Any, dst: Any, byte_count: Optional[int] = None) -> None:
        """
        Perform the actual data transfer.

        Args:
            src: Source object
            dst: Destination object
            byte_count: Optional byte count for DFB and pipe copies

        Raises:
            ValueError: If the transfer fails
        """
        ...

    def can_wait(self, src: Any, dst: Any, byte_count: Optional[int] = None) -> bool:
        """
        Check if wait() can proceed without blocking.

        Args:
            src: Source object
            dst: Destination object
            byte_count: Optional byte count for DFB and pipe copies

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


def _validate_byte_count_for_tensor(
    tensor: Tensor, byte_count: int, endpoint: str
) -> None:
    """Require addressable storage large enough for the requested byte range."""

    element_count = math.prod(tensor.padded_shape)
    capacity = tensor.size_in_bytes(element_count)
    if byte_count > capacity:
        raise ValueError(
            f"byte_count {byte_count} exceeds {endpoint} capacity {capacity}"
        )
    element_size = tensor.element_size
    if capacity != element_count * element_size:
        raise ValueError(
            "byte-counted simulation requires an element-addressable data type"
        )


def _require_dfb_block_format(block: Block, endpoint: str) -> ByteCopyFormat:
    """Use owning-DFB metadata because dry-run blocks contain sentinel tensors."""

    if block.dfb is None:
        raise ValueError(
            f"byte-counted {endpoint} block must be acquired from a dataflow buffer"
        )
    likeness = block.dfb.likeness_tensor
    if likeness.layout != TILE_LAYOUT:
        raise ValueError(
            f"byte-counted {endpoint} dataflow-buffer block must use TILE layout"
        )
    return ByteCopyFormat(layout=likeness.layout, dtype=likeness.dtype)


def _validate_byte_count_for_block(
    block: Block, byte_count: int, endpoint: str
) -> ByteCopyFormat:
    """Use declared DFB capacity because dry-run storage has zero elements."""

    block_format = _require_dfb_block_format(block, endpoint)
    assert block.dfb is not None
    capacity = block.dfb.capacity_bytes // block.dfb.block_count
    if byte_count > capacity:
        raise ValueError(
            f"byte_count {byte_count} exceeds {endpoint} capacity {capacity}"
        )
    if not _is_dry_run():
        _validate_byte_count_for_tensor(block.raw_tensor, byte_count, endpoint)
    return block_format


def _validate_byte_copy_formats(src: ByteCopyFormat, dst: ByteCopyFormat) -> None:
    """Allow different tile geometry without reinterpreting payload bytes."""

    if src.layout != dst.layout:
        raise ValueError(
            "byte-counted copy requires matching layouts; got "
            f"source {src.layout.name} and destination {dst.layout.name}"
        )
    if src.dtype != dst.dtype:
        raise ValueError(
            "byte-counted copy requires matching data types; got "
            f"source {src.dtype} and destination {dst.dtype}"
        )


def _validate_byte_copy_tensors(src: Tensor, dst: Tensor, byte_count: int) -> None:
    """Require element-addressable storage before copying individual bytes."""

    _validate_byte_copy_formats(
        ByteCopyFormat(layout=src.layout, dtype=src.dtype),
        ByteCopyFormat(layout=dst.layout, dtype=dst.dtype),
    )
    _validate_byte_count_for_tensor(src, byte_count, "source")
    _validate_byte_count_for_tensor(dst, byte_count, "destination")


def _copy_initial_tensor_bytes(src: Tensor, dst: Tensor, byte_count: int) -> None:
    """Copy exactly the initial byte range and preserve all later bytes."""

    _validate_byte_copy_tensors(src, dst, byte_count)
    complete_elements, partial_bytes = divmod(byte_count, src.element_size)
    src_values = src.to_torch().reshape(-1)
    dst_values = dst.to_torch().reshape(-1)
    dst_values[:complete_elements].copy_(src_values[:complete_elements])
    if partial_bytes == 0:
        return

    src_partial = src_values[complete_elements : complete_elements + 1].to(
        dtype=src.dtype
    )
    dst_partial = dst_values[complete_elements : complete_elements + 1].to(
        dtype=dst.dtype
    )
    src_partial_bytes = src_partial.view(dtype=torch.uint8).reshape(-1)
    dst_partial_bytes = dst_partial.view(dtype=torch.uint8).reshape(-1)
    dst_partial_bytes[:partial_bytes].copy_(src_partial_bytes[:partial_bytes])
    dst_values[complete_elements] = dst_partial.to(dtype=dst.underlying_dtype).item()


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
            def validate(self, src, dst, byte_count=None): ...
            def transfer(self, src, dst, byte_count=None): ...
    """

    def decorator(handler_cls: Type[CopyTransferHandler]):
        # Register handler in module-level registry
        HANDLER_REGISTRY[(src_type, dst_type)] = handler_cls()
        return handler_cls

    return decorator


@register_copy_handler(Block, Pipe)
class BlockToPipeHandler:
    """Handler for Block → Pipe (pipe send)."""

    def validate(
        self, src: Block, dst: AnyPipe, byte_count: Optional[int] = None
    ) -> None:
        """Validate the pipe send payload."""
        del dst
        if byte_count is not None:
            _validate_byte_count_for_block(src, byte_count, "source")

    def transfer(
        self, src: Block, dst: AnyPipe, byte_count: Optional[int] = None
    ) -> None:
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
            byte_count=byte_count,
            byte_copy_format=(
                _require_dfb_block_format(src, "source")
                if byte_count is not None
                else None
            ),
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
                byte_count=byte_count,
            )

    def can_wait(
        self, src: Block, dst: AnyPipe, byte_count: Optional[int] = None
    ) -> bool:
        """Block to Pipe copy completes immediately on wait()."""
        del src, dst, byte_count
        return True


@register_copy_handler(Tensor, Block)
class TensorToBlockHandler:
    """Handler for TTNN.Tensor -> Block transfers using tile-level indexing."""

    def validate(
        self, src: Tensor, dst: Block, byte_count: Optional[int] = None
    ) -> None:
        if byte_count is not None:
            raise ValueError("Tensor-to-block copy does not accept byte_count")
        _validate_tensor_to_block_shapes(
            src.layout, src.padded_shape, dst.layout, dst.shape
        )

    def transfer(
        self, src: Tensor, dst: Block, byte_count: Optional[int] = None
    ) -> None:
        """Transfer tensor data into Block."""
        del byte_count
        if _is_dry_run():
            return
        dst.copy_as_dest(src)

    def can_wait(
        self, src: Tensor, dst: Block, byte_count: Optional[int] = None
    ) -> bool:
        del src, dst, byte_count
        return True


@register_copy_handler(Block, Tensor)
class BlockToTensorHandler:
    """Handler for Block -> TTNN.Tensor transfers using tile-level indexing."""

    def validate(
        self, src: Block, dst: Tensor, byte_count: Optional[int] = None
    ) -> None:
        if byte_count is not None:
            raise ValueError("Block-to-tensor copy does not accept byte_count")
        _validate_block_to_tensor_shapes(
            src.layout, src.shape, dst.layout, dst.padded_shape
        )

    def transfer(
        self, src: Block, dst: Tensor, byte_count: Optional[int] = None
    ) -> None:
        """Transfer Block data into tensor."""
        del byte_count
        if _is_dry_run():
            return
        dst_raw = dst.to_torch()
        src_raw = src.raw_tensor.to_torch()
        dst_raw.copy_(src_raw.reshape(dst_raw.shape))

    def can_wait(
        self, src: Block, dst: Tensor, byte_count: Optional[int] = None
    ) -> bool:
        del src, dst, byte_count
        return True


@register_copy_handler(Block, Block)
class BlockToBlockHandler:
    """Handler for explicit byte-preserving copies between acquired blocks."""

    def validate(
        self, src: Block, dst: Block, byte_count: Optional[int] = None
    ) -> None:
        if byte_count is None:
            raise ValueError("Block-to-block copy requires byte_count")
        if src.acquisition != BlockAcquisition.WAIT:
            raise ValueError("Block-to-block copy source must come from DFB wait()")
        if dst.acquisition != BlockAcquisition.RESERVE:
            raise ValueError(
                "Block-to-block copy destination must come from DFB reserve()"
            )
        if src.dfb is None or dst.dfb is None or src.dfb is dst.dfb:
            raise ValueError(
                "Block-to-block copy requires distinct source and destination DFBs"
            )
        src_format = _validate_byte_count_for_block(src, byte_count, "source")
        dst_format = _validate_byte_count_for_block(dst, byte_count, "destination")
        _validate_byte_copy_formats(src_format, dst_format)

    def transfer(
        self, src: Block, dst: Block, byte_count: Optional[int] = None
    ) -> None:
        assert byte_count is not None
        if not _is_dry_run():
            _copy_initial_tensor_bytes(src.raw_tensor, dst.raw_tensor, byte_count)

    def can_wait(
        self, src: Block, dst: Block, byte_count: Optional[int] = None
    ) -> bool:
        del src, dst, byte_count
        return True


@register_copy_handler(Pipe, Block)
class PipeToBlockHandler:
    """Handler for Pipe → Block (pipe receive)."""

    def validate(
        self, src: AnyPipe, dst: Block, byte_count: Optional[int] = None
    ) -> None:
        """Validate the receiver's declared payload capacity."""
        del src
        if byte_count is not None:
            _validate_byte_count_for_block(dst, byte_count, "destination")

    def can_wait(
        self, src: AnyPipe, dst: Block, byte_count: Optional[int] = None
    ) -> bool:
        """Pipe to Block copy can only proceed when pipe has data for this node.

        Returns True only when there is at least one queued message that the
        current node has not yet received.  The greenlet scheduler polls this
        before calling transfer(), so transfer() can assume data is available.
        """
        del dst, byte_count
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

    def transfer(
        self, src: AnyPipe, dst: Block, byte_count: Optional[int] = None
    ) -> None:
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
                if message.byte_count != byte_count:
                    raise ValueError(
                        "Pipe sender and receiver must use the same byte_count; "
                        f"got sender {message.byte_count} and receiver {byte_count}"
                    )
                if byte_count is not None:
                    assert message.byte_copy_format is not None
                    _validate_byte_copy_formats(
                        message.byte_copy_format,
                        _require_dfb_block_format(dst, "destination"),
                    )
                if byte_count is None and message.grid_shape != dst.shape:
                    raise ValueError(
                        f"Destination Block shape {dst.shape} "
                        f"does not match pipe data shape {message.grid_shape}"
                    )

                # Payload copy only happens when data is present; in dry-run the
                # message carries no bytes (data is None) and the copy is skipped
                # while the queue bookkeeping below still runs.
                if message.data is not None:
                    if byte_count is None:
                        dst.copy_as_dest(message.data)
                    else:
                        _copy_initial_tensor_bytes(
                            message.data, dst.raw_tensor, byte_count
                        )

                if TRACE.enabled:
                    trace(
                        "pipe_recv",
                        pipe=get_pipe_name(src),
                        tiles=math.prod(message.grid_shape),
                        byte_count=byte_count,
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

    def validate(
        self,
        src: Block,
        dst: AnySrcPipeIdentity,
        byte_count: Optional[int] = None,
    ) -> None:
        self._get_delegate().validate(src, dst.pipe, byte_count)

    def transfer(
        self,
        src: Block,
        dst: AnySrcPipeIdentity,
        byte_count: Optional[int] = None,
    ) -> None:
        self._get_delegate().transfer(src, dst.pipe, byte_count)

    def can_wait(
        self,
        src: Block,
        dst: AnySrcPipeIdentity,
        byte_count: Optional[int] = None,
    ) -> bool:
        return self._get_delegate().can_wait(src, dst.pipe, byte_count)


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

    def validate(
        self,
        src: DstPipeIdentity,
        dst: Block,
        byte_count: Optional[int] = None,
    ) -> None:
        self._get_delegate().validate(src.pipe, dst, byte_count)

    def transfer(
        self,
        src: DstPipeIdentity,
        dst: Block,
        byte_count: Optional[int] = None,
    ) -> None:
        self._get_delegate().transfer(src.pipe, dst, byte_count)

    def can_wait(
        self,
        src: DstPipeIdentity,
        dst: Block,
        byte_count: Optional[int] = None,
    ) -> bool:
        return self._get_delegate().can_wait(src.pipe, dst, byte_count)
