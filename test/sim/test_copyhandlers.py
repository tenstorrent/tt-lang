# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from python.sim.copyhandlers import (
    handler_registry,
    TensorToBlockHandler,
    BlockToTensorHandler,
    BlockToMulticastHandler,
    MulticastToBlockHandler,
)
from python.sim.block import Block, Span
from python.sim.typedefs import MulticastAddress
from python.sim import torch_utils as tu
from python.sim.constants import TILE_SHAPE


class TestHandlerRegistry:
    """Test the handler registry mechanism."""

    def test_registry_populated(self):
        """Test that all handlers are registered."""
        assert (torch.Tensor, Block) in handler_registry
        assert (Block, torch.Tensor) in handler_registry
        assert (Block, MulticastAddress) in handler_registry
        assert (MulticastAddress, Block) in handler_registry

    def test_registry_handlers_correct_type(self):
        """Test that registered handlers are the correct instances."""
        assert isinstance(handler_registry[(torch.Tensor, Block)], TensorToBlockHandler)
        assert isinstance(handler_registry[(Block, torch.Tensor)], BlockToTensorHandler)
        assert isinstance(
            handler_registry[(Block, MulticastAddress)], BlockToMulticastHandler
        )
        assert isinstance(
            handler_registry[(MulticastAddress, Block)], MulticastToBlockHandler
        )


class TestTensorToBlockHandler:
    """Test TensorToBlockHandler validation and transfer."""

    def test_validate_success(self):
        """Test validation succeeds for matching shapes."""
        handler = TensorToBlockHandler()
        tensor = tu.ones(64, 32)  # 2 tiles
        buf: List[Optional[torch.Tensor]] = [None, None]
        block = Block(buf, 2, Span(0, 2))

        # Should not raise
        handler.validate(tensor, block)

    def test_validate_non_2d_tensor(self):
        """Test validation fails for non-2D tensors."""
        handler = TensorToBlockHandler()
        tensor = torch.ones(32, 32, 32)  # 3D tensor
        buf: List[Optional[torch.Tensor]] = [None]
        block = Block(buf, 1, Span(0, 1))

        with pytest.raises(ValueError, match="Copy only supports 2D tensors"):
            handler.validate(tensor, block)

    def test_validate_tile_count_mismatch(self):
        """Test validation fails when tile counts don't match."""
        handler = TensorToBlockHandler()
        tensor = tu.ones(96, 32)  # 3 tiles
        buf: List[Optional[torch.Tensor]] = [None, None]
        block = Block(buf, 2, Span(0, 2))

        with pytest.raises(
            ValueError, match="Tensor contains 3 tiles but Block has 2 slots"
        ):
            handler.validate(tensor, block)

    def test_transfer_single_tile(self):
        """Test transferring a single tile."""
        handler = TensorToBlockHandler()
        tensor = tu.full((32, 32), 42.0)
        buf: List[Optional[torch.Tensor]] = [None]
        block = Block(buf, 1, Span(0, 1))

        handler.transfer(tensor, block)

        assert block[0] is not None
        assert tu.allclose(block[0], tensor)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles."""
        handler = TensorToBlockHandler()
        # Create a 2x2 tile tensor (64x64)
        tensor = torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64)
        buf: List[Optional[torch.Tensor]] = [None, None, None, None]
        block = Block(buf, 4, Span(0, 4))

        handler.transfer(tensor, block)

        # Verify all tiles are populated
        for i in range(4):
            assert block[i] is not None
            assert block[i].shape == TILE_SHAPE


class TestBlockToTensorHandler:
    """Test BlockToTensorHandler validation and transfer."""

    def test_validate_success(self):
        """Test validation succeeds for matching shapes."""
        handler = BlockToTensorHandler()
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        block = Block(buf, 2, Span(0, 2))
        tensor = tu.zeros(64, 32)

        # Should not raise
        handler.validate(block, tensor)

    def test_validate_non_2d_tensor(self):
        """Test validation fails for non-2D tensors."""
        handler = BlockToTensorHandler()
        tile1 = tu.ones(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile1]
        block = Block(buf, 1, Span(0, 1))
        tensor = torch.zeros(32, 32, 32)  # 3D tensor

        with pytest.raises(ValueError, match="Copy only supports 2D tensors"):
            handler.validate(block, tensor)

    def test_validate_tile_count_mismatch(self):
        """Test validation fails when tile counts don't match."""
        handler = BlockToTensorHandler()
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        block = Block(buf, 2, Span(0, 2))
        tensor = tu.zeros(96, 32)  # 3 tiles

        with pytest.raises(ValueError, match="Expected 2 tiles but found 3"):
            handler.validate(block, tensor)

    def test_transfer_single_tile(self):
        """Test transferring a single tile."""
        handler = BlockToTensorHandler()
        tile = tu.full((32, 32), 3.14)
        buf: List[Optional[torch.Tensor]] = [tile]
        block = Block(buf, 1, Span(0, 1))
        tensor = tu.zeros(32, 32)

        handler.transfer(block, tensor)

        assert tu.allclose(tensor, tile)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles."""
        handler = BlockToTensorHandler()
        tile1 = tu.full((32, 32), 1.0)
        tile2 = tu.full((32, 32), 2.0)
        buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        block = Block(buf, 2, Span(0, 2))
        tensor = tu.zeros(64, 32)

        handler.transfer(block, tensor)

        # Verify tiles are combined correctly
        assert tu.allclose(tensor[0:32, :], tile1)
        assert tu.allclose(tensor[32:64, :], tile2)


class TestBlockToMulticastHandler:
    """Test BlockToMulticastHandler validation and transfer."""

    def test_validate_always_succeeds(self):
        """Test that validation always succeeds (no-op)."""
        handler = BlockToMulticastHandler()
        tile = tu.ones(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile]
        block = Block(buf, 1, Span(0, 1))
        mcast_addr = MulticastAddress(0, (1, 2))

        # Should not raise
        handler.validate(block, mcast_addr)

    def test_transfer_and_receive_single_tile(self):
        """Test multicast send and receive with single tile."""
        send_handler = BlockToMulticastHandler()
        recv_handler = MulticastToBlockHandler()

        # Sender prepares data
        tile = tu.full((32, 32), 42.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1))
        mcast_addr = MulticastAddress(0, (1,))

        # Send via multicast
        send_handler.transfer(src_block, mcast_addr)

        # Receiver retrieves data
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, dst_block)

        # Verify data was received correctly
        assert dst_block[0] is not None
        assert tu.allclose(dst_block[0], tile)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles via multicast."""
        send_handler = BlockToMulticastHandler()
        recv_handler = MulticastToBlockHandler()

        # Sender prepares data
        tile1 = tu.full((32, 32), 1.0)
        tile2 = tu.full((32, 32), 2.0)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2))
        mcast_addr = MulticastAddress(0, (1,))

        # Send via multicast
        send_handler.transfer(src_block, mcast_addr)

        # Receiver retrieves data
        dst_buf: List[Optional[torch.Tensor]] = [None, None]
        dst_block = Block(dst_buf, 2, Span(0, 2))
        recv_handler.transfer(mcast_addr, dst_block)

        # Verify data was received correctly
        assert tu.allclose(dst_block[0], tile1)
        assert tu.allclose(dst_block[1], tile2)


class TestMulticastToBlockHandler:
    """Test MulticastToBlockHandler validation and transfer."""

    def test_validate_always_succeeds(self):
        """Test that validation always succeeds (no-op)."""
        handler = MulticastToBlockHandler()
        mcast_addr = MulticastAddress(0, (1,))
        buf: List[Optional[torch.Tensor]] = [None]
        block = Block(buf, 1, Span(0, 1))

        # Should not raise
        handler.validate(mcast_addr, block)

    def test_transfer_timeout_no_data(self):
        """Test that transfer times out when no data is available."""
        recv_handler = MulticastToBlockHandler()
        # Use a unique address to avoid interference from other tests
        mcast_addr = MulticastAddress(99, (100,))
        buf: List[Optional[torch.Tensor]] = [None]
        block = Block(buf, 1, Span(0, 1))

        with pytest.raises(TimeoutError, match="Timeout waiting for multicast data"):
            recv_handler.transfer(mcast_addr, block)

    def test_transfer_success_single_receiver(self):
        """Test successful transfer with single receiver."""
        send_handler = BlockToMulticastHandler()
        recv_handler = MulticastToBlockHandler()
        mcast_addr = MulticastAddress(0, (1,))

        # Sender: send data
        tile = tu.full((32, 32), 99.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1))
        send_handler.transfer(src_block, mcast_addr)

        # Receiver: consume from multicast buffer
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, dst_block)

        # Verify data was transferred
        assert dst_block[0] is not None
        assert tu.allclose(dst_block[0], tile)

    def test_transfer_success_multiple_receivers(self):
        """Test successful transfer with multiple receivers."""
        send_handler = BlockToMulticastHandler()
        recv_handler = MulticastToBlockHandler()
        mcast_addr = MulticastAddress(0, (1, 2))

        # Sender: send data for 2 receivers
        tile = tu.full((32, 32), 77.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1))
        send_handler.transfer(src_block, mcast_addr)

        # First receiver
        buf1: List[Optional[torch.Tensor]] = [None]
        block1 = Block(buf1, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, block1)
        assert tu.allclose(block1[0], tile)

        # Second receiver
        buf2: List[Optional[torch.Tensor]] = [None]
        block2 = Block(buf2, 1, Span(0, 1))
        recv_handler.transfer(mcast_addr, block2)
        assert tu.allclose(block2[0], tile)

    def test_transfer_length_mismatch(self):
        """Test that transfer fails when Block length doesn't match data."""
        send_handler = BlockToMulticastHandler()
        recv_handler = MulticastToBlockHandler()
        mcast_addr = MulticastAddress(0, (1,))

        # Sender: send 2 tiles
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2))
        send_handler.transfer(src_block, mcast_addr)

        # Receiver: try to receive into 1-tile Block
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1))

        with pytest.raises(
            ValueError,
            match="Destination Block length .* does not match multicast data length",
        ):
            recv_handler.transfer(mcast_addr, dst_block)


if __name__ == "__main__":
    pytest.main([__file__])
