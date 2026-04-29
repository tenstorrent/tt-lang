# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy operation simulation for DataflowBuffer operations.

This module provides a simplified copy implementation for simulation purposes,
enabling data transfer operations between tensors and Blocks in the
DataflowBuffer system.
"""

from .dfb import Block
from .copyhandlers import (
    CopyEndpoint,
    CopyEndpointType,
    CopyTransferHandler,
    HANDLER_REGISTRY,
)
from .ttnnsim import Tensor, tile_count_from_tensor
from .sharding import try_count_locality
from .trace import trace
import math


def _copy_trace_fields(src: CopyEndpoint, dst: CopyEndpoint) -> dict:
    """Return extra fields for copy_start/copy_end when a Tensor is involved.

    When called from within a kernel (greenlet tagged with _sim_core), adds
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
    ):
        """
        Initialize a copy transaction from src to dst.

        Args:
            src: Source data (tensor, Block, or Pipe)
            dst: Destination (tensor, Block, or Pipe)

        Raises:
            ValueError: If the source and destination types are not supported
        """
        self._src = src
        self._dst = dst
        self._completed = False

        # Lookup and store the handler for this type combination
        handler = self._lookup_handler(type(src), type(dst))
        self._handler = handler

        # Mark blocks in state machine BEFORE validation - this transitions them to appropriate states
        # that prevent user access during the copy operation
        match src:
            case Block():
                src.mark_copy_as_source()
            case _:
                pass
        match dst:
            case Block():
                dst.mark_copy_as_dest()
            case _:
                pass

        # Validate immediately - let exceptions propagate to scheduler for context
        handler.validate(src, dst)

        trace(
            "copy_start",
            src=type(src).__name__,
            dst=type(dst).__name__,
            **_copy_trace_fields(src, dst),
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
        from .greenlet_scheduler import block_if_needed

        block_if_needed(self, "wait")

        # Transfer - let exceptions propagate to scheduler for context
        self._handler.transfer(self._src, self._dst)
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
    - Block → Pipe: Broadcast data to multiple cores (pipe send)
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
    return CopyTransaction(src, dst)
