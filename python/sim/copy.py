# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy operation simulation for CircularBuffer operations.

This module provides a simplified copy implementation for simulation purposes,
enabling data transfer operations between tensors and Blocks in the
CircularBuffer system.
"""

from typing import Any

from .block import Block
from .copyhandlers import (
    CopyEndpointType,
    CopyTransferHandler,
    handler_registry,
)

# Global registries to track Block objects locked by active copy operations
_read_locked: set = set()  # Blocks locked for reading (copy source)
_write_locked: set = set()  # Blocks locked for writing (copy destination)


def _extract_block(obj: Any) -> Any:
    """Extract the underlying Block from a context manager if applicable.

    Args:
        obj: Object that may be a Block or a context manager containing a Block

    Returns:
        The underlying Block if obj is a context manager, otherwise obj unchanged
    """
    # Check if object has a block() method (ReserveContext/WaitContext)
    if hasattr(obj, "block") and callable(obj.block):
        return obj.block()
    return obj


def check_can_read(obj: Block) -> None:
    """Check if a Block can be read from.

    Args:
        obj: Block to check

    Raises:
        RuntimeError: If Block is locked for writing by an active copy
    """
    if obj in _write_locked:
        raise RuntimeError(
            "Cannot read from Block: locked as copy destination until wait() completes"
        )


def check_can_write(obj: Block) -> None:
    """Check if a Block can be written to.

    Args:
        obj: Block to check

    Raises:
        RuntimeError: If Block is locked for reading or writing by an active copy
    """
    if obj in _read_locked:
        raise RuntimeError(
            "Cannot write to Block: locked as copy source until wait() completes"
        )
    if obj in _write_locked:
        raise RuntimeError(
            "Cannot write to Block: locked as copy destination until wait() completes"
        )


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
        src: Any,
        dst: Any,
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

        # Extract underlying Blocks from context managers if needed
        src_block = _extract_block(src)
        dst_block = _extract_block(dst)

        # Lookup and store the handler for this type combination
        handler = self._lookup_handler(type(src), type(dst))
        self._handler = handler

        # Validate immediately - let exceptions propagate to scheduler for context
        handler.validate(src, dst)

        # Check for locking conflicts before adding locks (Block only)
        if isinstance(src_block, Block):
            if src_block in _write_locked:
                raise RuntimeError(
                    "Cannot use Block as copy source: locked as copy destination until wait() completes"
                )
        if isinstance(dst_block, Block):
            if dst_block in _read_locked:
                raise RuntimeError(
                    "Cannot use Block as copy destination: locked as copy source until wait() completes"
                )
            if dst_block in _write_locked:
                raise RuntimeError(
                    "Cannot use Block as copy destination: already locked as copy destination until wait() completes"
                )

        # Lock Block source for reading and Block destination for writing
        if isinstance(src_block, Block):
            _read_locked.add(src_block)
        if isinstance(dst_block, Block):
            _write_locked.add(dst_block)

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
            return handler_registry[(src_type, dst_type)]
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

        # Extract underlying Blocks from context managers if needed
        src_block = _extract_block(self._src)
        dst_block = _extract_block(self._dst)

        # Unlock Block source and destination before transfer
        if isinstance(src_block, Block):
            _read_locked.discard(src_block)
        if isinstance(dst_block, Block):
            _write_locked.discard(dst_block)

        # Transfer - let exceptions propagate to scheduler for context
        self._handler.transfer(self._src, self._dst)
        self._completed = True

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


def copy(
    src: Any,
    dst: Any,
) -> CopyTransaction:
    """
    Create a copy transaction from source to destination.

    This function initiates a data transfer between the source and destination.
    The actual transfer occurs when wait() is called on the returned transaction.

    Supported transfer patterns:
    - torch.Tensor → Block: Load tensor data into circular buffer
    - Block → torch.Tensor: Extract tensor data from circular buffer
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
        # Transfer from tensor to circular buffer
        tx = copy(tensor_slice, cb_block)
        tx.wait()

        # Transfer from circular buffer to tensor
        tx = copy(cb_block, tensor_slice)
        tx.wait()
    """
    return CopyTransaction(src, dst)
