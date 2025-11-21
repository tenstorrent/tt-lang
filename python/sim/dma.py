# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DMA (Direct Memory Access) simulation for CircularBuffer operations.

This module provides a simplified DMA implementation for simulation purposes,
enabling data transfer operations between tensors and RingViews in the
CircularBuffer system.
"""

from typing import Any, Dict, List, Union
import torch
from . import torch_utils as tu
from .ringview import RingView
from .constants import TILE_SHAPE
from .typedefs import MulticastAddress


# Global multicast buffer for simulating NoC multicast communication
# In a real implementation, this would be handled by the NoC hardware
# TODO: This "List[Any]" should be made more specific, should need refactoring
_multicast_buffer: Dict[MulticastAddress, List[Any]] = {}


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
        # Validate supported type combinations immediately
        if isinstance(src, torch.Tensor) and isinstance(dst, RingView):
            # tensor → RingView: supported
            pass
        elif isinstance(src, RingView) and isinstance(dst, torch.Tensor):
            # RingView → tensor: supported
            pass
        elif isinstance(src, RingView) and isinstance(dst, MulticastAddress):
            # RingView → MulticastAddress: multicast send (supported)
            pass
        elif isinstance(src, MulticastAddress) and isinstance(dst, RingView):
            # MulticastAddress → RingView: multicast receive (supported)
            pass
        else:
            # Unsupported type combination
            src_type = type(src).__name__
            dst_type = type(dst).__name__
            raise ValueError(
                f"Unsupported DMA transfer from {src_type} to {dst_type}. "
                f"Supported transfers: tensor↔RingView, RingView→MulticastAddress, MulticastAddress→RingView"
            )

        self._src = src
        self._dst = dst
        self._completed = False

    # In this simulation context, we elect to do the dma transfer right at the
    # wait() call to render the wait() call meaningful. The alternatives would
    # be to do the transfer at the time of dma() call (which would make wait() a
    # no-op) or to spawn a background thread to do the transfer asynchronously
    # (which would complicate the simulation unnecessarily).
    def wait(self) -> None:
        """
        Wait for the DMA transaction to complete.

        In this simulation, the transfer is performed immediately when wait()
        is called. In a real implementation, this would block until the
        asynchronous transfer completes.

        Raises:
            ValueError: If the transfer operation fails
        """
        if not self._completed:
            # In simulation, we perform the copy immediately
            # Type combinations are already validated in __init__
            if isinstance(self._src, torch.Tensor) and isinstance(self._dst, RingView):
                # Copying from tensor to RingView - split tensor into individual tiles
                try:
                    # Validate that we're dealing with 2D tensors
                    if len(self._src.shape) != 2:
                        raise ValueError(
                            f"DMA only supports 2D tensors, got {len(self._src.shape)}D tensor with shape {self._src.shape}"
                        )

                    # Calculate total number of tiles in the tensor
                    num_tiles = tu.tile_count(self._src.shape, TILE_SHAPE)
                    expected_tiles = len(self._dst)

                    if num_tiles != expected_tiles:
                        raise ValueError(
                            f"Tensor contains {num_tiles} tiles but RingView has {expected_tiles} slots"
                        )

                    # Calculate dimensions needed for row-major indexing
                    width_tiles = self._src.shape[1] // TILE_SHAPE[1]

                    # Extract tiles in row-major order
                    for tile_idx in range(num_tiles):
                        # Convert linear index to 2D tile coordinates
                        h_tile = tile_idx // width_tiles
                        w_tile = tile_idx % width_tiles

                        # Calculate tensor slice coordinates
                        start_row = h_tile * TILE_SHAPE[0]
                        end_row = (h_tile + 1) * TILE_SHAPE[0]
                        start_col = w_tile * TILE_SHAPE[1]
                        end_col = (w_tile + 1) * TILE_SHAPE[1]

                        tile = self._src[start_row:end_row, start_col:end_col]
                        self._dst[tile_idx] = tile

                except Exception as e:
                    raise ValueError(f"Failed to transfer tensor to RingView: {e}")
            elif isinstance(self._src, RingView) and isinstance(
                self._dst, torch.Tensor
            ):
                # Copying from RingView to tensor - combine individual tiles into one tensor
                try:
                    # Validate that we're dealing with 2D tensors
                    if len(self._dst.shape) != 2:
                        raise ValueError(
                            f"DMA only supports 2D tensors, got {len(self._dst.shape)}D tensor with shape {self._dst.shape}"
                        )

                    dst_tiles = tu.tile_count(self._dst.shape, TILE_SHAPE)
                    if len(self._src) != dst_tiles:
                        raise ValueError(
                            f"Expected {len(self._src)} tiles but found {dst_tiles}"
                        )

                    # Calculate dimensions needed for row-major indexing
                    width_tiles = self._dst.shape[1] // TILE_SHAPE[1]

                    # Reconstruct tensor by placing tiles in their proper 2D positions
                    for tile_idx in range(dst_tiles):
                        # Convert linear index to 2D tile coordinates
                        h_tile = tile_idx // width_tiles
                        w_tile = tile_idx % width_tiles

                        start_row = h_tile * TILE_SHAPE[0]
                        end_row = (h_tile + 1) * TILE_SHAPE[0]
                        start_col = w_tile * TILE_SHAPE[1]
                        end_col = (w_tile + 1) * TILE_SHAPE[1]

                        tile = self._src[tile_idx]
                        self._dst[start_row:end_row, start_col:end_col] = tile

                except Exception as e:
                    raise ValueError(f"Failed to transfer RingView to tensor: {e}")
            elif isinstance(self._src, RingView) and isinstance(
                self._dst, MulticastAddress
            ):
                # Multicast send: RingView → MulticastAddress
                # Store the data in a shared multicast buffer accessible by all cores
                try:
                    src_data = [self._src[i] for i in range(len(self._src))]
                    # Store in a global multicast buffer keyed by the multicast address
                    # In a real implementation, this would send packets over the NoC
                    _multicast_buffer[self._dst] = src_data
                except Exception as e:
                    raise ValueError(f"Failed to multicast send: {e}")
            elif isinstance(self._src, MulticastAddress) and isinstance(
                self._dst, RingView
            ):
                # Multicast receive: MulticastAddress → RingView
                # Retrieve the data from the shared multicast buffer
                try:
                    # Poll until data is available in the multicast buffer
                    # In a real implementation, this would be hardware-level blocking
                    import time

                    timeout = 2.0  # 2 second timeout to detect deadlocks
                    start_time = time.time()
                    while self._src not in _multicast_buffer:
                        if time.time() - start_time > timeout:
                            raise TimeoutError(
                                f"Timeout waiting for multicast data. "
                                f"The sender may not have called dma(ringview, mcast_addr).wait() "
                                f"or there may be a deadlock."
                            )
                        time.sleep(0.001)  # Small sleep to avoid busy-waiting

                    src_data = _multicast_buffer[self._src]
                    if len(self._dst) != len(src_data):
                        raise ValueError(
                            f"Destination RingView length ({len(self._dst)}) "
                            f"does not match multicast data length ({len(src_data)})"
                        )
                    self._dst.store(src_data)
                except TimeoutError:
                    raise
                except Exception as e:
                    raise ValueError(f"Failed to multicast receive: {e}")
            # Note: No else case needed since types are validated in __init__
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
