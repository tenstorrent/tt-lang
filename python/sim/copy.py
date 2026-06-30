# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy operation simulation for DataflowBuffer operations.

This module provides a simplified copy implementation for simulation purposes,
enabling data transfer operations between tensors and Blocks in the
DataflowBuffer system.
"""

import math
import sys
from typing import Optional, Tuple

from .context import get_context
from .copyhandlers import (
    CopyEndpoint,
    CopyEndpointType,
    CopyTransferHandler,
    HANDLER_REGISTRY,
)
from .dfb import Block
from .greenlet_scheduler import block_if_needed
from .sharding import try_count_locality
from .trace import TRACE, trace
from .ttnnsim import Tensor, tile_count_from_tensor
from .pipe import Pipe, SrcPipeIdentity


def _copy_trace_fields(src: CopyEndpoint, dst: CopyEndpoint) -> dict:
    """Return extra fields for copy_start/copy_end when a Tensor is involved.

    When called from within a kernel (greenlet tagged with _sim_node), adds
    element-level locality fields: local_l1, remote_l1, dram.
    """
    match (src, dst):
        case (Tensor(), Block()):
            tensor, direction, tiles = src, "read", tile_count_from_tensor(src)
        case (Block(), Tensor()):
            tensor, direction, tiles = dst, "write", math.prod(src.shape)
        case _:
            return {}

    fields: dict = {
        "tensor": getattr(tensor, "_name", None) or type(tensor).__name__,
        "tiles": tiles,
        "direction": direction,
    }
    locality = try_count_locality(tensor)
    if locality is not None:
        local_elems, remote_elems, dram_elems = locality
        # Convert element counts to tile counts using the same ratio as `tiles`.
        # For TILE_LAYOUT: elements_per_tile = prod(shape) / tile_count.
        # For ROW_MAJOR_LAYOUT: elements_per_tile = 1 (each element is a unit).
        # Integer division is exact for standard tile-aligned sharding.
        total_elems = math.prod(tensor.shape)
        if total_elems > 0:
            fields["local_l1"] = local_elems * tiles // total_elems
            fields["remote_l1"] = remote_elems * tiles // total_elems
            fields["dram"] = dram_elems * tiles // total_elems
    return fields


class CopyTransaction:
    """
    Represents a copy transaction that can be waited on.

    This is a simplified mock implementation for simulation purposes.
    In a real implementation, this would handle asynchronous data transfers
    between different memory locations or devices.

    Example:
        tx = copy(source_tensor, destination_block)
        tx.wait()  # Wait for transfer to complete
    """

    def __init__(
        self,
        src: CopyEndpoint,
        dst: CopyEndpoint,
        user_location: Optional[Tuple[str, int]] = None,
    ):
        """
        Initialize a copy transaction from src to dst.

        Args:
            src: Source data (tensor, Block, or Pipe)
            dst: Destination (tensor, Block, or Pipe)
            user_location: Pre-captured ``(filename, lineno)`` for the user
                code initiating this copy.  Passed in by :func:`copy` so that
                ``Block.mark_copy_as_{source,dest}`` can skip the per-call
                stack walk in :func:`find_user_code_location`.  ``None`` only
                from tests that construct ``CopyTransaction`` directly.

        Raises:
            ValueError: If the source and destination types are not supported
        """
        self._src = src
        self._dst = dst
        self._completed = False
        self._transfer_performed = False
        # Stable trace label, computed once at construction so the scheduler
        # can read it via direct attribute access from inside the hot
        # block_current_kernel path instead of paying for a getattr/fallback.
        self._trace_name: str = f"tx_{id(self) & 0xFFFF:04x}"

        # Lookup and store the handler for this type combination
        handler = self._lookup_handler(type(src), type(dst))
        self._handler = handler

        # Mark blocks in state machine BEFORE validation - this transitions them to appropriate states
        # that prevent user access during the copy operation
        match src:
            case Block():
                src.mark_copy_as_source(user_location)
            case _:
                pass
        match dst:
            case Block():
                dst.mark_copy_as_dest(user_location)
            case _:
                pass

        # Validate immediately - let exceptions propagate to scheduler for context
        handler.validate(src, dst)

        if TRACE.enabled:
            trace(
                "copy_start",
                src=type(src).__name__,
                dst=type(dst).__name__,
                **_copy_trace_fields(src, dst),
            )

        if self._starts_on_copy():
            self._handler.transfer(self._src, self._dst)
            self._transfer_performed = True

    def _starts_on_copy(self) -> bool:
        """Return true for transfers whose side effects begin at copy()."""
        return isinstance(self._src, Block) and isinstance(
            self._dst, (Pipe, SrcPipeIdentity)
        )

    @staticmethod
    def _lookup_handler(
        src_type: CopyEndpointType, dst_type: CopyEndpointType
    ) -> CopyTransferHandler:
        """
        Look up the handler for a given (src_type, dst_type) pair.

        Args:
            src_type: Source type class (must be a valid copy endpoint type)
            dst_type: Destination type class (must be a valid copy endpoint type)

        Returns:
            The registered handler for this type combination

        Raises:
            ValueError: If no handler is registered for this type combination
        """
        try:
            return HANDLER_REGISTRY[(src_type, dst_type)]
        except KeyError:
            raise ValueError(
                f"No copy handler registered for ({src_type.__name__}, {dst_type.__name__})"
            ) from None

    def wait(self) -> None:
        """
        Wait for the copy transaction to complete.

        In this simulation, the transfer is performed immediately when wait()
        is called by delegating to the registered handler's transfer() method.
        In a real implementation, this would block until the asynchronous
        transfer completes.

        Raises:
            ValueError: If the transfer operation fails
        """
        if self._completed:
            return

        # Block if copy cannot proceed
        block_if_needed(self, "wait")

        # Transfer - let exceptions propagate to scheduler for context.
        if not self._transfer_performed:
            # Each handler decides what to do in dry-run mode: payload-only
            # handlers (Tensor<->Block) skip the byte copy, while structural
            # handlers (Pipe send/receive) still maintain their queue
            # bookkeeping so pipe sequencing is exercised symmetrically. The
            # block state transitions below always fire so structural checks
            # (state machine, deadlock) remain fully exercised.
            self._handler.transfer(self._src, self._dst)
            self._transfer_performed = True
        self._completed = True

        # Mark tx.wait() complete in state machine - this transitions blocks back to accessible states
        match self._src:
            case Block():
                self._src.mark_tx_wait_complete()
            case _:
                pass
        match self._dst:
            case Block():
                self._dst.mark_tx_wait_complete()
            case _:
                pass

        if TRACE.enabled:
            trace(
                "copy_end",
                src=type(self._src).__name__,
                dst=type(self._dst).__name__,
                **_copy_trace_fields(self._src, self._dst),
            )

    def can_wait(self) -> bool:
        """
        Check if wait() can proceed without blocking.

        The semantics depend on the copy type:
        - Tensor ↔ Block: Always returns True (synchronous transfer)
        - Block → Pipe: Always returns True (completes immediately)
        - Pipe → Block: Returns True only when pipe has data available

        Returns:
            True if wait() can proceed without blocking
        """
        return self._handler.can_wait(self._src, self._dst)

    @property
    def is_completed(self) -> bool:
        """Check if the copy transaction has completed."""
        return self._completed


class GroupTransfer:
    """Group of transfer handles that can be waited on together.

    Collects handles returned by ttl.copy and waits for all of them at once
    via wait_all().  No further add() calls are permitted after wait_all().

    Example:
        gxf = GroupTransfer()
        for dst in destinations:
            gxf.add(ttl.copy(src_blk, dst))
        gxf.wait_all()
    """

    def __init__(self) -> None:
        self._transfers: list[CopyTransaction] = []
        self._waited: bool = False

    def add(self, xf: CopyTransaction) -> None:
        """Add a transfer handle to the group.

        Raises:
            RuntimeError: If called after wait_all().
        """
        if self._waited:
            raise RuntimeError("GroupTransfer.add() called after wait_all()")
        self._transfers.append(xf)

    def wait_all(self) -> None:
        """Wait for all transfers in the group to complete.

        Raises:
            RuntimeError: If called more than once.
        """
        if self._waited:
            raise RuntimeError("GroupTransfer.wait_all() called more than once")
        self._waited = True
        for xf in self._transfers:
            xf.wait()


def copy(
    src: CopyEndpoint,
    dst: CopyEndpoint,
) -> CopyTransaction:
    """
    Create a copy transaction from source to destination.

    This function initiates a data transfer between the source and destination.
    The actual transfer occurs when wait() is called on the returned transaction.

    Supported transfer patterns:
    - torch.Tensor → Block: Load tensor data into dataflow buffer
    - Block → torch.Tensor: Extract tensor data from dataflow buffer
    - Block → Pipe: Broadcast data to multiple nodes (pipe send)
    - Pipe → Block: Receive broadcasted data from pipe (pipe receive)

    Args:
        src: Source data (tensor, Block, or Pipe)
        dst: Destination (tensor, Block, or Pipe)

    Returns:
        CopyTransaction object that can be waited on

    Raises:
        ValueError: Immediately if unsupported type combinations are provided

    Example:
        # Transfer from tensor to dataflow buffer
        tx = copy(tensor_slice, dfb_block)
        tx.wait()

        # Transfer from dataflow buffer to tensor
        tx = copy(dfb_block, tensor_slice)
        tx.wait()
    """
    # Capture the user's source location ONCE here (cheap: one ``_getframe``,
    # no chain walk) so that ``Block.mark_copy_as_{source,dest}`` -- both
    # invoked inside ``CopyTransaction.__init__`` below -- can stash it without
    # paying for two ``find_user_code_location()`` stack walks per copy.  In
    # the matmul-tutorial dry-run that's 4.2 M walks eliminated.
    #
    # ``sys._getframe(1)`` returns the immediate caller of ``copy()``.  For the
    # public ``ttl.copy`` entry point that's user code (or a user-defined
    # helper, which ``find_user_code_location`` would also surface since both
    # are non-simulator frames).  Reused immediately below for the auto-wait
    # call-site lookup.
    frame = sys._getframe(1)
    user_location: Tuple[str, int] = (frame.f_code.co_filename, frame.f_lineno)

    handle = CopyTransaction(src, dst, user_location=user_location)

    # Case A: bare ttl.copy(...) with no assignment — auto-wait immediately.
    # The AST analysis in analyze_kernel_function identifies these call sites
    # and registers their (caller_code, abs_lineno) in context.auto_wait_copy_lines.
    # Using equality-based set lookup so that code objects from different files
    # with identical bodies are still matched correctly.
    ctx = get_context()
    if (
        ctx.auto_wait_copy_lines
        and (frame.f_code, frame.f_lineno) in ctx.auto_wait_copy_lines
    ):
        handle.wait()

    return handle
