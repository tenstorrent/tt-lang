"""
DMA (Direct Memory Access) simulation for CircularBuffer operations.

This module provides a simplified DMA implementation for simulation purposes,
enabling data transfer operations between tensors and RingViews in the 
CircularBuffer system.
"""

from typing import Union, List
import torch
from .ringview import RingView
from .constants import TILE_SIZE


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
    
    def __init__(self, src: Union[torch.Tensor, RingView[torch.Tensor]], 
                 dst: Union[torch.Tensor, RingView[torch.Tensor]]):
        """
        Initialize a DMA transaction from src to dst.
        
        Args:
            src: Source data (tensor or RingView)
            dst: Destination (tensor or RingView)
            
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
        else:
            # Unsupported type combination
            src_type = type(src).__name__
            dst_type = type(dst).__name__
            raise ValueError(
                f"Unsupported DMA transfer from {src_type} to {dst_type}. "
                f"Only tensor↔RingView transfers are supported."
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
                    # Calculate number of tiles: tensor height / TILE_SIZE
                    num_tiles = self._src.shape[0] // TILE_SIZE
                    expected_tiles = len(self._dst)
                    
                    if num_tiles != expected_tiles:
                        raise ValueError(f"Tensor contains {num_tiles} tiles but RingView has {expected_tiles} slots")
                    
                    # Split tensor into individual tiles and place each in a RingView slot
                    for i in range(num_tiles):
                        start_row = i * TILE_SIZE
                        end_row = (i + 1) * TILE_SIZE
                        tile = self._src[start_row:end_row, :]  # Extract one tile
                        self._dst[i] = tile
                        
                except Exception as e:
                    raise ValueError(f"Failed to transfer tensor to RingView: {e}")
            elif isinstance(self._src, RingView) and isinstance(self._dst, torch.Tensor):
                # Copying from RingView to tensor - combine individual tiles into one tensor
                try:
                    # Collect all tiles from RingView and stack them vertically
                    tiles: List[torch.Tensor] = []
                    for i in range(len(self._src)):
                        tile = self._src[i]
                        tiles.append(tile)
                    
                    if len(tiles) != len(self._src):
                        raise ValueError(f"Expected {len(self._src)} tiles but found {len(tiles)}")
                    
                    # Stack tiles vertically to reconstruct the original tensor
                    reconstructed_tensor = torch.cat(tiles, dim=0) # type: ignore
                    
                    if reconstructed_tensor.shape != self._dst.shape:
                        raise ValueError(f"Reconstructed tensor shape {reconstructed_tensor.shape} doesn't match destination {self._dst.shape}")
                    
                    # Copy reconstructed tensor to destination
                    self._dst[:] = reconstructed_tensor
                    
                except Exception as e:
                    raise ValueError(f"Failed to transfer RingView to tensor: {e}")
            # Note: No else case needed since types are validated in __init__
            self._completed = True
    
    @property
    def completed(self) -> bool:
        """Check if the DMA transaction has completed."""
        return self._completed


def dma(src: Union[torch.Tensor, RingView[torch.Tensor]], 
        dst: Union[torch.Tensor, RingView[torch.Tensor]]) -> DMATransaction:
    """
    Create a DMA transaction from source to destination.
    
    This function initiates a DMA transfer between the source and destination.
    The actual transfer occurs when wait() is called on the returned transaction.
    
    Supported transfer patterns:
    - torch.Tensor → RingView: Load tensor data into circular buffer
    - RingView → torch.Tensor: Extract tensor data from circular buffer
    
    Args:
        src: Source data (tensor or RingView)
        dst: Destination (tensor or RingView)
        
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