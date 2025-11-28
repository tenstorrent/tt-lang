# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DMA (Direct Memory Access) simulation for CircularBuffer operations.

This module provides a simplified DMA implementation for simulation purposes,
enabling data transfer operations between tensors and RingViews in the
CircularBuffer system.
"""

from typing import Any, Type, Union
import torch
from .ringview import RingView
from .typedefs import MulticastAddress
from .dmahandlers import DMATransferHandler, handler_registry


class DMATransaction:
    """
    Represents a DMA transaction that can be waited on.

    This is a simplified mock implementation for simulation purposes.
    In a real implementation, this would handle asynchronous data transfers
    between different memory locations or devices.

    Example:
        tx = dma(source_tensor, destination_ringview)
        tx.wait()  # Wait for transfer to complete
    """

    def __init__(
        self,
        src: Union[torch.Tensor, RingView[torch.Tensor], MulticastAddress],
        dst: Union[torch.Tensor, RingView[torch.Tensor], MulticastAddress],
    ):
        """
        Initialize a DMA transaction from src to dst.

        Args:
            src: Source data (tensor, RingView, or MulticastAddress)
            dst: Destination (tensor, RingView, or MulticastAddress)

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
                f"Unsupported or invalid DMA transfer from {src_type} to {dst_type}: {e}"
            ) from e

    @staticmethod
    def _lookup_handler(src_type: Type[Any], dst_type: Type[Any]) -> DMATransferHandler:
        """
        Look up the handler for a given (src_type, dst_type) pair.

        Args:
            src_type: Source type class
            dst_type: Destination type class

        Returns:
            The registered handler for this type combination

        Raises:
            ValueError: If no handler is registered for this type combination
        """
        try:
            return handler_registry[(src_type, dst_type)]
        except KeyError:
            raise ValueError(
                f"No DMA handler registered for ({src_type.__name__}, {dst_type.__name__})"
            )

    def wait(self) -> None:
        """
        Wait for the DMA transaction to complete.

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
            raise ValueError(f"DMA transfer failed: {e}") from e

        self._completed = True

    @property
    def completed(self) -> bool:
        """Check if the DMA transaction has completed."""
        return self._completed


def dma(
    src: Union[torch.Tensor, RingView[torch.Tensor], MulticastAddress],
    dst: Union[torch.Tensor, RingView[torch.Tensor], MulticastAddress],
) -> DMATransaction:
    """
    Create a DMA transaction from source to destination.

    This function initiates a DMA transfer between the source and destination.
    The actual transfer occurs when wait() is called on the returned transaction.

    Supported transfer patterns:
    - torch.Tensor → RingView: Load tensor data into circular buffer
    - RingView → torch.Tensor: Extract tensor data from circular buffer
    - RingView → MulticastAddress: Broadcast data to multiple cores (multicast send)
    - MulticastAddress → RingView: Receive broadcasted data from multicast (multicast receive)

    Args:
        src: Source data (tensor, RingView, or MulticastAddress)
        dst: Destination (tensor, RingView, or MulticastAddress)

    Returns:
        DMATransaction object that can be waited on

    Raises:
        ValueError: Immediately if unsupported type combinations are provided

    Example:
        # Transfer from tensor to circular buffer
        tx = dma(tensor_slice, cb_ringview)
        tx.wait()

        # Transfer from circular buffer to tensor
        tx = dma(cb_ringview, tensor_slice)
        tx.wait()
    """
    return DMATransaction(src, dst)
