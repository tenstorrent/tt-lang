# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from python.sim.dmahandlers import (
    handler_registry,
    TensorToRingViewHandler,
    RingViewToTensorHandler,
    RingViewToMulticastHandler,
    MulticastToRingViewHandler,
)
from python.sim.ringview import RingView, Span
from python.sim.typedefs import MulticastAddress, MulticastType
from python.sim import torch_utils as tu
from python.sim.constants import TILE_SHAPE


class TestHandlerRegistry:
    """Test the handler registry mechanism."""

    def test_registry_populated(self):
        """Test that all handlers are registered."""
        assert (torch.Tensor, RingView) in handler_registry
        assert (RingView, torch.Tensor) in handler_registry
        assert (RingView, MulticastAddress) in handler_registry
        assert (MulticastAddress, RingView) in handler_registry

    def test_registry_handlers_correct_type(self):
        """Test that registered handlers are the correct instances."""
        assert isinstance(
            handler_registry[(torch.Tensor, RingView)], TensorToRingViewHandler
        )
        assert isinstance(
            handler_registry[(RingView, torch.Tensor)], RingViewToTensorHandler
        )
        assert isinstance(
            handler_registry[(RingView, MulticastAddress)], RingViewToMulticastHandler
        )
        assert isinstance(
            handler_registry[(MulticastAddress, RingView)], MulticastToRingViewHandler
        )


class TestTensorToRingViewHandler:
    """Test TensorToRingViewHandler validation and transfer."""

    def test_validate_success(self):
        """Test validation succeeds for matching shapes."""
        handler = TensorToRingViewHandler()
        tensor = tu.ones(64, 32)  # 2 tiles
        buf: List[Optional[torch.Tensor]] = [None, None]
        ringview = RingView(buf, 2, Span(0, 2))

        # Should not raise
        handler.validate(tensor, ringview)

    def test_validate_non_2d_tensor(self):
        """Test validation fails for non-2D tensors."""
        handler = TensorToRingViewHandler()
        tensor = torch.ones(32, 32, 32)  # 3D tensor
        buf: List[Optional[torch.Tensor]] = [None]
        ringview = RingView(buf, 1, Span(0, 1))

        with pytest.raises(ValueError, match="DMA only supports 2D tensors"):
            handler.validate(tensor, ringview)

    def test_validate_tile_count_mismatch(self):
        """Test validation fails when tile counts don't match."""
        handler = TensorToRingViewHandler()
        tensor = tu.ones(96, 32)  # 3 tiles
        buf: List[Optional[torch.Tensor]] = [None, None]
        ringview = RingView(buf, 2, Span(0, 2))

        with pytest.raises(
            ValueError, match="Tensor contains 3 tiles but RingView has 2 slots"
        ):
            handler.validate(tensor, ringview)

    def test_transfer_single_tile(self):
        """Test transferring a single tile."""
        handler = TensorToRingViewHandler()
        tensor = tu.full((32, 32), 42.0)
        buf: List[Optional[torch.Tensor]] = [None]
        ringview = RingView(buf, 1, Span(0, 1))

        handler.transfer(tensor, ringview)

        assert ringview[0] is not None
        assert tu.allclose(ringview[0], tensor)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles."""
        handler = TensorToRingViewHandler()
        # Create a 2x2 tile tensor (64x64)
        tensor = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
        buf: List[Optional[torch.Tensor]] = [None, None, None, None]
        ringview = RingView(buf, 4, Span(0, 4))

        handler.transfer(tensor, ringview)

        # Verify all tiles are populated
        for i in range(4):
            assert ringview[i] is not None
            assert ringview[i].shape == TILE_SHAPE


class TestRingViewToTensorHandler:
    """Test RingViewToTensorHandler validation and transfer."""

    def test_validate_success(self):
        """Test validation succeeds for matching shapes."""
        handler = RingViewToTensorHandler()
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        ringview = RingView(buf, 2, Span(0, 2))
        tensor = tu.zeros(64, 32)

        # Should not raise
        handler.validate(ringview, tensor)

    def test_validate_non_2d_tensor(self):
        """Test validation fails for non-2D tensors."""
        handler = RingViewToTensorHandler()
        tile1 = tu.ones(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile1]
        ringview = RingView(buf, 1, Span(0, 1))
        tensor = torch.zeros(32, 32, 32)  # 3D tensor

        with pytest.raises(ValueError, match="DMA only supports 2D tensors"):
            handler.validate(ringview, tensor)

    def test_validate_tile_count_mismatch(self):
        """Test validation fails when tile counts don't match."""
        handler = RingViewToTensorHandler()
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        ringview = RingView(buf, 2, Span(0, 2))
        tensor = tu.zeros(96, 32)  # 3 tiles

        with pytest.raises(ValueError, match="Expected 2 tiles but found 3"):
            handler.validate(ringview, tensor)

    def test_transfer_single_tile(self):
        """Test transferring a single tile."""
        handler = RingViewToTensorHandler()
        tile = tu.full((32, 32), 3.14)
        buf: List[Optional[torch.Tensor]] = [tile]
        ringview = RingView(buf, 1, Span(0, 1))
        tensor = tu.zeros(32, 32)

        handler.transfer(ringview, tensor)

        assert tu.allclose(tensor, tile)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles."""
        handler = RingViewToTensorHandler()
        tile1 = tu.full((32, 32), 1.0)
        tile2 = tu.full((32, 32), 2.0)
        buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        ringview = RingView(buf, 2, Span(0, 2))
        tensor = tu.zeros(64, 32)

        handler.transfer(ringview, tensor)

        # Verify tiles are combined correctly
        assert tu.allclose(tensor[0:32, :], tile1)
        assert tu.allclose(tensor[32:64, :], tile2)


class TestRingViewToMulticastHandler:
    """Test RingViewToMulticastHandler validation and transfer."""

    def test_validate_always_succeeds(self):
        """Test that validation always succeeds (no-op)."""
        handler = RingViewToMulticastHandler()
        tile = tu.ones(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile]
        ringview = RingView(buf, 1, Span(0, 1))
        mcast_addr = MulticastAddress(MulticastType.PUSH, (0, 1, 2))

        # Should not raise
        handler.validate(ringview, mcast_addr)

    def test_transfer_and_receive_single_tile(self):
        """Test multicast send and receive with single tile."""
        send_handler = RingViewToMulticastHandler()
        recv_handler = MulticastToRingViewHandler()

        # Sender prepares data
        tile = tu.full((32, 32), 42.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_ringview = RingView(src_buf, 1, Span(0, 1))
        mcast_addr = MulticastAddress(MulticastType.PUSH, (0, 1))

        # Send via multicast
        send_handler.transfer(src_ringview, mcast_addr)

        # Receiver retrieves data
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ringview = RingView(dst_buf, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, dst_ringview)

        # Verify data was received correctly
        assert dst_ringview[0] is not None
        assert tu.allclose(dst_ringview[0], tile)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles via multicast."""
        send_handler = RingViewToMulticastHandler()
        recv_handler = MulticastToRingViewHandler()

        # Sender prepares data
        tile1 = tu.full((32, 32), 1.0)
        tile2 = tu.full((32, 32), 2.0)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_ringview = RingView(src_buf, 2, Span(0, 2))
        mcast_addr = MulticastAddress(MulticastType.PUSH, (0, 1))

        # Send via multicast
        send_handler.transfer(src_ringview, mcast_addr)

        # Receiver retrieves data
        dst_buf: List[Optional[torch.Tensor]] = [None, None]
        dst_ringview = RingView(dst_buf, 2, Span(0, 2))
        recv_handler.transfer(mcast_addr, dst_ringview)

        # Verify data was received correctly
        assert tu.allclose(dst_ringview[0], tile1)
        assert tu.allclose(dst_ringview[1], tile2)


class TestMulticastToRingViewHandler:
    """Test MulticastToRingViewHandler validation and transfer."""

    def test_validate_always_succeeds(self):
        """Test that validation always succeeds (no-op)."""
        handler = MulticastToRingViewHandler()
        mcast_addr = MulticastAddress(MulticastType.PUSH, (0, 1))
        buf: List[Optional[torch.Tensor]] = [None]
        ringview = RingView(buf, 1, Span(0, 1))

        # Should not raise
        handler.validate(mcast_addr, ringview)

    def test_transfer_timeout_no_data(self):
        """Test that transfer times out when no data is available."""
        recv_handler = MulticastToRingViewHandler()
        # Use a unique address to avoid interference from other tests
        mcast_addr = MulticastAddress(MulticastType.PUSH, (99, 100))
        buf: List[Optional[torch.Tensor]] = [None]
        ringview = RingView(buf, 1, Span(0, 1))

        with pytest.raises(TimeoutError, match="Timeout waiting for multicast data"):
            recv_handler.transfer(mcast_addr, ringview)

    def test_transfer_success_single_receiver(self):
        """Test successful transfer with single receiver."""
        send_handler = RingViewToMulticastHandler()
        recv_handler = MulticastToRingViewHandler()
        mcast_addr = MulticastAddress(MulticastType.PUSH, (0, 1))

        # Sender: send data
        tile = tu.full((32, 32), 99.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_ringview = RingView(src_buf, 1, Span(0, 1))
        send_handler.transfer(src_ringview, mcast_addr)

        # Receiver: consume from multicast buffer
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ringview = RingView(dst_buf, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, dst_ringview)

        # Verify data was transferred
        assert dst_ringview[0] is not None
        assert tu.allclose(dst_ringview[0], tile)

    def test_transfer_success_multiple_receivers(self):
        """Test successful transfer with multiple receivers."""
        send_handler = RingViewToMulticastHandler()
        recv_handler = MulticastToRingViewHandler()
        mcast_addr = MulticastAddress(MulticastType.PUSH, (0, 1, 2))

        # Sender: send data for 2 receivers
        tile = tu.full((32, 32), 77.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_ringview = RingView(src_buf, 1, Span(0, 1))
        send_handler.transfer(src_ringview, mcast_addr)

        # First receiver
        buf1: List[Optional[torch.Tensor]] = [None]
        ringview1 = RingView(buf1, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, ringview1)
        assert tu.allclose(ringview1[0], tile)

        # Second receiver
        buf2: List[Optional[torch.Tensor]] = [None]
        ringview2 = RingView(buf2, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, ringview2)
        assert tu.allclose(ringview2[0], tile)

    def test_transfer_length_mismatch(self):
        """Test that transfer fails when RingView length doesn't match data."""
        send_handler = RingViewToMulticastHandler()
        recv_handler = MulticastToRingViewHandler()
        mcast_addr = MulticastAddress(MulticastType.PUSH, (0, 1))

        # Sender: send 2 tiles
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_ringview = RingView(src_buf, 2, Span(0, 2))
        send_handler.transfer(src_ringview, mcast_addr)

        # Receiver: try to receive into 1-tile RingView
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ringview = RingView(dst_buf, 1, Span(0, 1))

        with pytest.raises(
            ValueError,
            match="Destination RingView length .* does not match multicast data length",
        ):
            recv_handler.transfer(mcast_addr, dst_ringview)


if __name__ == "__main__":
    pytest.main([__file__])
