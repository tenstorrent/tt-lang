# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
from test_utils import (
    make_arange_tensor,
    make_full_tile,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
    tensors_equal,
)

from python.sim import ttnn
from python.sim.block import Block, Span
from python.sim.cbstate import CBSlot
from python.sim.constants import TILE_SHAPE
from python.sim.copyhandlers import (
    BlockToPipeHandler,
    BlockToTensorHandler,
    PipeToBlockHandler,
    TensorToBlockHandler,
    handler_registry,
)
from python.sim.typedefs import Pipe


class TestHandlerRegistry:
    """Test the handler registry mechanism."""

    def test_registry_populated(self):
        """Test that all handlers are registered."""
        assert (ttnn.Tensor, Block) in handler_registry
        assert (Block, ttnn.Tensor) in handler_registry
        assert (Block, Pipe) in handler_registry
        assert (Pipe, Block) in handler_registry

    def test_registry_handlers_correct_type(self):
        """Test that registered handlers are the correct instances."""
        assert isinstance(handler_registry[(ttnn.Tensor, Block)], TensorToBlockHandler)
        assert isinstance(handler_registry[(Block, ttnn.Tensor)], BlockToTensorHandler)
        assert isinstance(handler_registry[(Block, Pipe)], BlockToPipeHandler)
        assert isinstance(handler_registry[(Pipe, Block)], PipeToBlockHandler)


class TestTensorToBlockHandler:
    """Test TensorToBlockHandler validation and transfer."""

    def test_validate_success(self):
        """Test validation succeeds for matching shapes."""
        handler = TensorToBlockHandler()
        tensor = make_rand_tensor(64, 32)  # 2 tiles
        buf: List[CBSlot] = [None, None]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        # Should not raise
        handler.validate(tensor, block)

    def test_validate_non_2d_tensor(self):
        """Test validation fails for non-2D tensors."""
        import torch

        handler = TensorToBlockHandler()
        # Create a 3D torch tensor and wrap it
        torch_3d = torch.ones(32, 32, 32)
        tensor = ttnn.Tensor(torch_3d)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        with pytest.raises(ValueError, match="Copy only supports 2D tensors"):
            handler.validate(tensor, block)

    def test_validate_tile_count_mismatch(self):
        """Test validation fails when tile counts don't match."""
        handler = TensorToBlockHandler()
        tensor = make_rand_tensor(96, 32)  # 3 tiles
        buf: List[CBSlot] = [None, None]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        with pytest.raises(
            ValueError, match="Tensor shape.*does not match.*Block shape"
        ):
            handler.validate(tensor, block)

    def test_transfer_single_tile(self):
        """Test transferring a single tile."""
        handler = TensorToBlockHandler()
        tensor = make_full_tile(42.0)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        handler.transfer(tensor, block)

        assert block[0] is not None
        assert tensors_equal(block[0], tensor)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles."""
        handler = TensorToBlockHandler()
        # Create a 2x2 tile tensor (64x64)
        tensor = make_arange_tensor(64, 64)
        buf: List[CBSlot] = [None, None, None, None]
        block = Block(buf, 4, Span(0, 4), shape=(2, 2))

        handler.transfer(tensor, block)

        # Verify all tiles are populated and have correct tile shape (1x1 in tile coords)
        for i in range(4):
            assert block[i] is not None
            # Each tile is a ttnn.Tensor with shape (32, 32) in elements
            assert block[i].shape == TILE_SHAPE


class TestBlockToTensorHandler:
    """Test BlockToTensorHandler validation and transfer."""

    def test_validate_success(self):
        """Test validation succeeds for matching shapes."""
        handler = BlockToTensorHandler()
        tile1 = make_ones_tile()
        tile2 = make_zeros_tile()
        buf: List[CBSlot] = [tile1, tile2]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))
        tensor = make_rand_tensor(64, 32)

        # Should not raise
        handler.validate(block, tensor)

    def test_validate_non_2d_tensor(self):
        """Test validation fails for non-2D tensors."""
        import torch

        handler = BlockToTensorHandler()
        tile1 = make_ones_tile()
        buf: List[CBSlot] = [tile1]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))
        # Create a 3D torch tensor and wrap it
        torch_3d = torch.zeros(32, 32, 32)
        tensor = ttnn.Tensor(torch_3d)

        with pytest.raises(ValueError, match="Copy only supports 2D tensors"):
            handler.validate(block, tensor)

    def test_validate_tile_count_mismatch(self):
        """Test validation fails when tile counts don't match."""
        handler = BlockToTensorHandler()
        tile1 = make_ones_tile()
        tile2 = make_zeros_tile()
        buf: List[CBSlot] = [tile1, tile2]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))
        tensor = make_rand_tensor(96, 32)  # 3 tiles

        with pytest.raises(
            ValueError, match="Tensor shape.*does not match.*Block shape"
        ):
            handler.validate(block, tensor)

    def test_transfer_single_tile(self):
        """Test transferring a single tile."""
        handler = BlockToTensorHandler()
        tile = make_full_tile(3.14)
        buf: List[CBSlot] = [tile]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))
        tensor = make_zeros_tile()

        handler.transfer(block, tensor)

        assert tensors_equal(tensor, tile)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles."""
        handler = BlockToTensorHandler()
        tile1 = make_full_tile(1.0)
        tile2 = make_full_tile(2.0)
        buf: List[CBSlot] = [tile1, tile2]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))
        tensor = make_rand_tensor(64, 32)

        handler.transfer(block, tensor)

        # Verify tiles are combined correctly using tile-level indexing
        dest_tile1 = tensor[0:1, 0:1]
        dest_tile2 = tensor[1:2, 0:1]
        assert tensors_equal(dest_tile1, tile1)
        assert tensors_equal(dest_tile2, tile2)


class TestBlockToPipeHandler:
    """Test BlockToPipeHandler validation and transfer."""

    def test_validate_always_succeeds(self):
        """Test that validation always succeeds (no-op)."""
        handler = BlockToPipeHandler()
        tile = make_ones_tile()
        buf: List[CBSlot] = [tile]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))
        pipe = Pipe(0, (1, 2))

        # Should not raise
        handler.validate(block, pipe)

    def test_transfer_and_receive_single_tile(self):
        """Test pipe send and receive with single tile."""
        send_handler = BlockToPipeHandler()
        recv_handler = PipeToBlockHandler()

        # Sender prepares data
        tile = make_full_tile(42.0)
        src_buf: List[CBSlot] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1), shape=(1, 1))
        pipe = Pipe(0, 1)

        # Send via pipe
        send_handler.transfer(src_block, pipe)

        # Receiver retrieves data
        dst_buf: List[CBSlot] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1), shape=(1, 1))
        recv_handler.transfer(pipe, dst_block)

        # Verify data was received correctly
        assert dst_block[0] is not None
        assert tensors_equal(dst_block[0], tile)

    def test_transfer_multiple_tiles(self):
        """Test transferring multiple tiles via pipe."""
        send_handler = BlockToPipeHandler()
        recv_handler = PipeToBlockHandler()

        # Sender prepares data
        tile1 = make_full_tile(1.0)
        tile2 = make_full_tile(2.0)
        src_buf: List[CBSlot] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2), shape=(2, 1))
        pipe = Pipe(0, 1)

        # Send via pipe
        send_handler.transfer(src_block, pipe)

        # Receiver retrieves data
        dst_buf: List[CBSlot] = [None, None]
        dst_block = Block(dst_buf, 2, Span(0, 2), shape=(2, 1))
        recv_handler.transfer(pipe, dst_block)

        # Verify data was received correctly
        assert dst_block[0] is not None
        assert dst_block[1] is not None
        assert tensors_equal(dst_block[0], tile1)
        assert tensors_equal(dst_block[1], tile2)


class TestPipeToBlockHandler:
    """Test PipeToBlockHandler validation and transfer."""

    def test_validate_always_succeeds(self):
        """Test that validation always succeeds (no-op)."""
        handler = PipeToBlockHandler()
        pipe = Pipe(0, 1)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        # Should not raise
        handler.validate(pipe, block)

    def test_transfer_timeout_no_data(self):
        """Test that transfer times out when no data is available."""
        recv_handler = PipeToBlockHandler()
        # Use a unique address to avoid interference from other tests
        pipe = Pipe(99, 100)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        with pytest.raises(TimeoutError, match="Timeout waiting for pipe data"):
            recv_handler.transfer(pipe, block)

    def test_transfer_success_single_receiver(self):
        """Test successful transfer with single receiver."""
        send_handler = BlockToPipeHandler()
        recv_handler = PipeToBlockHandler()
        pipe = Pipe(0, 1)

        # Sender: send data
        tile = make_full_tile(99.0)
        src_buf: List[CBSlot] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1), shape=(1, 1))
        send_handler.transfer(src_block, pipe)

        # Receiver: consume from pipe buffer
        dst_buf: List[CBSlot] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1), shape=(1, 1))
        recv_handler.transfer(pipe, dst_block)

        # Verify data was transferred
        assert dst_block[0] is not None
        assert tensors_equal(dst_block[0], tile)

    def test_transfer_success_multiple_receivers(self):
        """Test successful transfer with multiple receivers."""
        send_handler = BlockToPipeHandler()
        recv_handler = PipeToBlockHandler()
        # Rectangular range from (0,1) to (0,2) covers 2 cores
        pipe = Pipe((0, 0), ((0, 1), (0, 2)))

        # Sender: send data for 2 receivers
        tile = make_full_tile(77.0)
        src_buf: List[CBSlot] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1), shape=(1, 1))
        send_handler.transfer(src_block, pipe)

        # First receiver
        buf1: List[CBSlot] = [None]
        block1 = Block(buf1, 1, Span(0, 1), shape=(1, 1))
        recv_handler.transfer(pipe, block1)
        assert block1[0] is not None
        assert tensors_equal(block1[0], tile)

        # Second receiver
        buf2: List[CBSlot] = [None]
        block2 = Block(buf2, 1, Span(0, 1), shape=(1, 1))
        recv_handler.transfer(pipe, block2)
        assert block2[0] is not None
        assert tensors_equal(block2[0], tile)

    def test_transfer_length_mismatch(self):
        """Test that transfer fails when Block length doesn't match data."""
        send_handler = BlockToPipeHandler()
        recv_handler = PipeToBlockHandler()
        # Single core unicast
        pipe = Pipe((0, 0), (0, 1))

        # Sender: send 2 tiles
        tile1 = make_ones_tile()
        tile2 = make_zeros_tile()
        src_buf: List[CBSlot] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2), shape=(2, 1))
        send_handler.transfer(src_block, pipe)

        # Receiver: try to receive into 1-tile Block
        dst_buf: List[CBSlot] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1), shape=(1, 1))

        with pytest.raises(
            ValueError,
            match="Destination Block length .* does not match pipe data length",
        ):
            recv_handler.transfer(pipe, dst_block)


if __name__ == "__main__":
    pytest.main([__file__])
