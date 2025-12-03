# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Copy operation simulation for CircularBuffer operations.

This module provides a simplified copy implementation for simulation purposes,
enabling data transfer operations between tensors and Blocks in the
CircularBuffer system.
"""

from .copyhandlers import (
    CopyTransferHandler,
    CopyEndpoint,
    CopyEndpointType,
    handler_registry,
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
        src: CopyEndpoint,
        dst: CopyEndpoint,
    ):
        """
        Initialize a copy transaction from src to dst.

        Args:
            src: Source data (tensor, Block, or MulticastAddress)
            dst: Destination (tensor, Block, or MulticastAddress)

        Raises:
            ValueError: If the source and destination types are not supported
        """
        self._src = src
        self._dst = dst
        self._completed = False

        # Lookup and store the handler for this type combination
        handler = self._lookup_handler(type(src), type(dst))
        self._handler = handler

        # Validate immediately
        try:
            handler.validate(src, dst)
        except Exception as e:
            src_type = type(src).__name__
            dst_type = type(dst).__name__
            raise ValueError(
                f"Unsupported or invalid copy transfer from {src_type} to {dst_type}: {e}"
            ) from e

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
            )

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

        try:
            self._handler.transfer(self._src, self._dst)
        except Exception as e:
            raise ValueError(f"Copy transfer failed: {e}") from e

        self._completed = True

    @property
    def is_completed(self) -> bool:
        """Check if the copy transaction has completed."""
        return self._completed


def copy(
    src: CopyEndpoint,
    dst: CopyEndpoint,
) -> CopyTransaction:
    """
    Create a copy transaction from source to destination.

    This function initiates a data transfer between the source and destination.
    The actual transfer occurs when wait() is called on the returned transaction.

    Supported transfer patterns:
    - torch.Tensor → Block: Load tensor data into circular buffer
    - Block → torch.Tensor: Extract tensor data from circular buffer
    - Block → MulticastAddress: Broadcast data to multiple cores (multicast send)
    - MulticastAddress → Block: Receive broadcasted data from multicast (multicast receive)

    Args:
        src: Source data (tensor, Block, or MulticastAddress)
        dst: Destination (tensor, Block, or MulticastAddress)

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
