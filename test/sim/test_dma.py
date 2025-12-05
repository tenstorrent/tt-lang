# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for DMA (Direct Memory Access) simulation.

Tests the DMA transfer functionality between tensors and RingViews,
including error handling and edge cases.
"""

import pytest
import torch
from python.sim import torch_utils as tu
from python.sim.dma import DMATransaction, dma
from python.sim.ringview import RingView, Span
from python.sim.constants import TILE_SHAPE
from typing import Optional, List
from python.sim.typedefs import MulticastAddress, MulticastType


class TestDMATransaction:
    """Test DMATransaction class functionality."""

    def test_dma_transaction_creation_tensor_to_ringview(self) -> None:
        """Test creating a DMA transaction from tensor to RingView."""
        tensor = tu.randn(64, 32)  # 2x1 tile block
        buf: List[Optional[torch.Tensor]] = [None, None]
        ringview = RingView(buf, 2, Span(0, 2))

        tx = DMATransaction(tensor, ringview)
        assert not tx.completed

    def test_dma_transaction_creation_ringview_to_tensor(self) -> None:
        """Test creating a DMA transaction from RingView to tensor."""
        buf: List[Optional[torch.Tensor]] = [tu.ones(32, 32), tu.zeros(32, 32)]
        ringview = RingView(buf, 2, Span(0, 2))
        tensor = tu.zeros(64, 32)  # 2x1 tile block

        tx = DMATransaction(ringview, tensor)
        assert not tx.completed

    def test_dma_transaction_unsupported_types(self) -> None:
        """Test that unsupported type combinations raise ValueError."""
        tensor1 = tu.randn(32, 32)
        tensor2 = tu.zeros(32, 32)

        # tensor → tensor not supported
        with pytest.raises(
            ValueError, match="No DMA handler registered for \\(Tensor, Tensor\\)"
        ):
            DMATransaction(tensor1, tensor2)

        # RingView → RingView not supported
        buf: List[Optional[torch.Tensor]] = [None, None]
        ringview1 = RingView(buf, 2, Span(0, 2))
        ringview2 = RingView(buf, 2, Span(0, 2))
        with pytest.raises(
            ValueError, match="No DMA handler registered for \\(RingView, RingView\\)"
        ):
            DMATransaction(ringview1, ringview2)


class TestTensorToRingViewDMA:
    """Test DMA operations from tensor to RingView."""

    def test_transfer_single_tile_to_ringview(self) -> None:
        """Test transferring a single tile tensor to RingView."""
        source = tu.ones(32, 32)  # Single tile
        buf: List[Optional[torch.Tensor]] = [None]  # Single slot for single tile
        ringview = RingView(buf, 1, Span(0, 1))

        tx = dma(source, ringview)
        tx.wait()

        # Verify completion
        assert tx.completed

        # Verify data was transferred correctly
        assert tu.allclose(ringview[0], source)

    def test_transfer_multi_tile_to_ringview(self) -> None:
        """Test transferring a multi-tile tensor to RingView."""
        # Create 2x1 tile tensor
        tile0 = tu.ones(32, 32)
        tile1 = tu.full((32, 32), 2.0)
        source = torch.cat([tile0, tile1], dim=0)  # type: ignore  # 64x32 tensor

        buf: List[Optional[torch.Tensor]] = [None, None, None, None]
        ringview = RingView(buf, 4, Span(0, 2))

        tx = dma(source, ringview)
        tx.wait()

        assert tx.completed
        assert tu.allclose(ringview[0], tile0)
        assert tu.allclose(ringview[1], tile1)

    def test_transfer_mismatched_tile_count(self) -> None:
        """Test that mismatched tile count raises ValueError."""
        # 3 tiles in tensor but 2 slots in RingView
        source = tu.ones(96, 32)  # 3 tiles
        buf: List[Optional[torch.Tensor]] = [None, None, None]
        ringview = RingView(buf, 3, Span(0, 2))  # Only 2 slots

        with pytest.raises(
            ValueError, match="Tensor contains 3 tiles but RingView has 2 slots"
        ):
            tx = dma(source, ringview)  # type: ignore

    def test_transfer_to_single_slot_ringview(self) -> None:
        """Test transferring to a RingView with a single slot."""
        source = tu.full((32, 32), 42.0)
        buf: List[Optional[torch.Tensor]] = [None]
        ringview = RingView(buf, 1, Span(0, 1))

        tx = dma(source, ringview)
        tx.wait()

        assert tx.completed
        assert tu.allclose(ringview[0], source)


class TestRingViewToTensorDMA:
    """Test DMA operations from RingView to tensor."""

    def test_transfer_single_tile_from_ringview(self) -> None:
        """Test transferring a single tile from RingView to tensor."""
        tile0 = tu.full((32, 32), 3.14)
        tile1 = tu.full((32, 32), 2.71)
        buf: List[Optional[torch.Tensor]] = [tile0, tile1]
        ringview = RingView(buf, 2, Span(0, 2))

        destination = tu.zeros(64, 32)

        tx = dma(ringview, destination)
        tx.wait()

        assert tx.completed
        # Check that tiles were stacked correctly
        expected = torch.cat([tile0, tile1], dim=0)  # type: ignore
        assert tu.allclose(destination, expected)

    def test_transfer_multi_tile_from_ringview(self) -> None:
        """Test transferring multiple tiles from RingView to tensor."""
        tiles = [tu.full((32, 32), float(i)) for i in range(4)]
        buf = tiles  # type: ignore  # Don't use Optional typing to avoid assignment error
        ringview = RingView(buf, 4, Span(0, 4))  # type: ignore

        destination = tu.zeros(128, 32)  # 4 tiles

        tx = dma(ringview, destination)
        tx.wait()

        assert tx.completed
        # Verify each tile was placed correctly
        for i in range(4):
            start_row = i * TILE_SHAPE[0]
            end_row = (i + 1) * TILE_SHAPE[0]
            tile_slice = destination[start_row:end_row, :]
            assert tu.allclose(tile_slice, tiles[i])

    def test_transfer_shape_mismatch(self) -> None:
        """Test that shape mismatch between RingView and tensor raises ValueError."""
        tile0 = tu.ones(32, 32)
        tile1 = tu.zeros(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile0, tile1]
        ringview = RingView(buf, 2, Span(0, 2))

        # Wrong destination shape
        destination = tu.zeros(96, 32)  # 3 tiles, but RingView has 2

        with pytest.raises(ValueError, match="Expected 2 tiles but found 3"):
            tx = dma(ringview, destination)  # type: ignore

    def test_transfer_from_single_slot_ringview(self) -> None:
        """Test transferring from a RingView with a single slot."""
        tile = tu.full((32, 32), 9.99)
        buf: List[Optional[torch.Tensor]] = [None]
        ringview = RingView(buf, 1, Span(0, 1))
        ringview[0] = tile

        destination = tu.zeros(32, 32)

        tx = dma(ringview, destination)
        tx.wait()

        assert tx.completed
        assert tu.allclose(destination, tile)


class TestDMAConvenienceFunction:
    """Test the dma() convenience function."""

    def test_dma_function_creates_transaction(self) -> None:
        """Test that dma() function creates and returns a DMATransaction."""
        source = tu.randn(64, 32)  # 2 tiles to match RingView size
        buf: List[Optional[torch.Tensor]] = [None, None]
        ringview = RingView(buf, 2, Span(0, 2))

        tx = dma(source, ringview)

        match tx:
            case DMATransaction():
                assert not tx.completed
            case _:
                raise AssertionError(
                    f"Expected DMATransaction, got {type(tx).__name__}"
                )


class TestDMAComplexOperations:
    """Test complex DMA operation scenarios."""

    def test_multiple_sequential_transfers(self) -> None:
        """Test performing multiple DMA transfers in sequence."""
        # Setup: Create source tensors
        tensor1 = tu.full((64, 32), 1.0)  # 2 tiles

        # Intermediate RingView buffer
        buf: List[Optional[torch.Tensor]] = [None, None, None, None]
        ringview = RingView(buf, 4, Span(0, 2))

        # Stage 1: Load first tensor to RingView
        tx1 = dma(tensor1, ringview)
        tx1.wait()
        assert tx1.completed

        # Stage 2: Process tiles in RingView (simulate some operation)
        processed_tiles = [ringview[i] * 10.0 for i in range(len(ringview))]
        ringview2 = RingView(processed_tiles, 2, Span(0, 2))  # type: ignore

        # Stage 3: Extract processed data back to tensor
        result = tu.zeros(64, 32)
        tx2 = dma(ringview2, result)
        tx2.wait()
        assert tx2.completed

        # Verify the transformation
        expected = tensor1 * 10.0
        assert tu.allclose(result, expected)

    def test_dma_with_single_element_ringview(self) -> None:
        """Test DMA with minimal RingView (single element)."""
        source = tu.full((32, 32), 123.456)
        buf: List[Optional[torch.Tensor]] = [None]
        ringview = RingView(buf, 1, Span(0, 1))

        # Transfer to RingView
        tx1 = dma(source, ringview)
        tx1.wait()

        # Transfer back to tensor
        destination = tu.zeros(32, 32)
        tx2 = dma(ringview, destination)
        tx2.wait()

        assert tu.allclose(destination, source)


class TestDMAErrorHandling:
    """Test DMA error conditions and edge cases."""

    def test_dma_with_empty_ringview(self) -> None:
        """Test DMA behavior with zero-length RingView."""
        source = tu.ones(32, 32)
        buf: List[Optional[torch.Tensor]] = []
        ringview = RingView(buf, 0, Span(0, 0))

        # Should fail when trying to create DMA to empty RingView
        with pytest.raises(ValueError):
            tx = dma(source, ringview)  # type: ignore


class TestMulticastDMA:
    """Tests for multicast DMA using the public `dma` API."""

    def test_multicast_single_tile_single_receiver(self) -> None:
        """Send a single tile via multicast and receive it."""
        tile = tu.full((32, 32), 123.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_ringview = RingView(src_buf, 1, Span(0, 1))

        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ringview = RingView(dst_buf, 1, Span(0, 1))

        mcast_addr = MulticastAddress(MulticastType.PUSH, (210, 211))

        tx_send = dma(src_ringview, mcast_addr)
        tx_send.wait()

        tx_recv = dma(mcast_addr, dst_ringview)
        tx_recv.wait()

        assert dst_ringview[0] is not None
        assert tu.allclose(dst_ringview[0], tile)

    def test_multicast_multiple_tiles_multiple_receivers(self) -> None:
        """Send multiple tiles and have multiple receivers consume them."""
        tile1 = tu.full((32, 32), 1.0)
        tile2 = tu.full((32, 32), 2.0)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_ringview = RingView(src_buf, 2, Span(0, 2))

        mcast_addr = MulticastAddress(MulticastType.PUSH, (211, 212, 213))

        tx_send = dma(src_ringview, mcast_addr)
        tx_send.wait()

        # First receiver
        dst1: List[Optional[torch.Tensor]] = [None, None]
        dst_ring1 = RingView(dst1, 2, Span(0, 2))
        tx_r1 = dma(mcast_addr, dst_ring1)
        tx_r1.wait()
        assert tu.allclose(dst_ring1[0], tile1)
        assert tu.allclose(dst_ring1[1], tile2)

        # Second receiver
        dst2: List[Optional[torch.Tensor]] = [None, None]
        dst_ring2 = RingView(dst2, 2, Span(0, 2))
        tx_r2 = dma(mcast_addr, dst_ring2)
        tx_r2.wait()
        assert tu.allclose(dst_ring2[0], tile1)
        assert tu.allclose(dst_ring2[1], tile2)

    def test_multicast_length_mismatch_raises(self) -> None:
        """Receiver with mismatched length should raise ValueError at wait()."""
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_ringview = RingView(src_buf, 2, Span(0, 2))

        mcast_addr = MulticastAddress(MulticastType.PUSH, (212, 213))

        tx_send = dma(src_ringview, mcast_addr)
        tx_send.wait()

        # Receiver with wrong length
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ring = RingView(dst_buf, 1, Span(0, 1))
        tx_recv = dma(mcast_addr, dst_ring)
        with pytest.raises(
            ValueError,
            match="Destination RingView length .* does not match multicast data length",
        ):
            tx_recv.wait()

    def test_multicast_receive_timeout(self) -> None:
        """Receiving on an address with no send should timeout."""
        mcast_addr = MulticastAddress(MulticastType.PUSH, (99, 100))
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ring = RingView(dst_buf, 1, Span(0, 1))

        tx_recv = dma(mcast_addr, dst_ring)
        with pytest.raises(ValueError, match="Timeout waiting for multicast data"):
            tx_recv.wait()
