# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for copy operation simulation.

Tests the copy transfer functionality between tensors and Blocks,
including error handling and edge cases.
"""

import pytest
import torch
from python.sim import torch_utils as tu
from python.sim.copy import CopyTransaction, copy
from python.sim.block import Block, Span
from python.sim.constants import TILE_SHAPE
from typing import Optional, List
from python.sim.typedefs import Pipe


class TestCopyTransaction:
    """Test CopyTransaction class functionality."""

    def test_copy_transaction_creation_tensor_to_block(self) -> None:
        """Test creating a copy transaction from tensor to Block."""
        tensor = tu.randn(64, 32)  # 2x1 tile block
        buf: List[Optional[torch.Tensor]] = [None, None]
        block = Block(buf, 2, Span(0, 2))

        tx = CopyTransaction(tensor, block)
        assert not tx.is_completed

    def test_copy_transaction_creation_block_to_tensor(self) -> None:
        """Test creating a copy transaction from Block to tensor."""
        buf: List[Optional[torch.Tensor]] = [tu.ones(32, 32), tu.zeros(32, 32)]
        block = Block(buf, 2, Span(0, 2))
        tensor = tu.zeros(64, 32)  # 2x1 tile block

        tx = CopyTransaction(block, tensor)
        assert not tx.is_completed

    def test_copy_transaction_unsupported_types(self) -> None:
        """Test that unsupported type combinations raise ValueError."""
        tensor1 = tu.randn(32, 32)
        tensor2 = tu.zeros(32, 32)

        # tensor → tensor not supported
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Tensor, Tensor\\)"
        ):
            CopyTransaction(tensor1, tensor2)

        # Block → Block not supported
        buf: List[Optional[torch.Tensor]] = [None, None]
        block1 = Block(buf, 2, Span(0, 2))
        block2 = Block(buf, 2, Span(0, 2))
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Block, Block\\)"
        ):
            CopyTransaction(block1, block2)


class TestTensorToBlockCopy:
    """Test copy operations from tensor to Block."""

    def test_transfer_single_tile_to_block(self) -> None:
        """Test transferring a single tile tensor to Block."""
        source = tu.ones(32, 32)  # Single tile
        buf: List[Optional[torch.Tensor]] = [None]  # Single slot for single tile
        block = Block(buf, 1, Span(0, 1))

        tx = copy(source, block)
        tx.wait()

        # Verify completion
        assert tx.is_completed

        # Verify data was transferred correctly
        assert tu.allclose(block[0], source)

    def test_transfer_multi_tile_to_block(self) -> None:
        """Test transferring a multi-tile tensor to Block."""
        # Create 2x1 tile tensor
        tile0 = tu.ones(32, 32)
        tile1 = tu.full((32, 32), 2.0)
        source = torch.cat([tile0, tile1], dim=0)  # type: ignore  # 64x32 tensor

        buf: List[Optional[torch.Tensor]] = [None, None, None, None]
        block = Block(buf, 4, Span(0, 2))

        tx = copy(source, block)
        tx.wait()

        assert tx.is_completed
        assert tu.allclose(block[0], tile0)
        assert tu.allclose(block[1], tile1)

    def test_transfer_mismatched_tile_count(self) -> None:
        """Test that mismatched tile count raises ValueError."""
        # 3 tiles in tensor but 2 slots in Block
        source = tu.ones(96, 32)  # 3 tiles
        buf: List[Optional[torch.Tensor]] = [None, None, None]
        block = Block(buf, 3, Span(0, 2))  # Only 2 slots

        with pytest.raises(
            ValueError, match="Tensor contains 3 tiles but Block has 2 slots"
        ):
            tx = copy(source, block)  # type: ignore

    def test_transfer_to_single_slot_block(self) -> None:
        """Test transferring to a Block with a single slot."""
        source = tu.full((32, 32), 42.0)
        buf: List[Optional[torch.Tensor]] = [None]
        block = Block(buf, 1, Span(0, 1))

        tx = copy(source, block)
        tx.wait()

        assert tx.is_completed
        assert tu.allclose(block[0], source)


class TestBlockToTensorCopy:
    """Test copy operations from Block to tensor."""

    def test_transfer_single_tile_from_block(self) -> None:
        """Test transferring a single tile from Block to tensor."""
        tile0 = tu.full((32, 32), 3.14)
        tile1 = tu.full((32, 32), 2.71)
        buf: List[Optional[torch.Tensor]] = [tile0, tile1]
        block = Block(buf, 2, Span(0, 2))

        destination = tu.zeros(64, 32)

        tx = copy(block, destination)
        tx.wait()

        assert tx.is_completed
        # Check that tiles were stacked correctly
        expected = torch.cat([tile0, tile1], dim=0)  # type: ignore
        assert tu.allclose(destination, expected)

    def test_transfer_multi_tile_from_block(self) -> None:
        """Test transferring multiple tiles from Block to tensor."""
        tiles = [tu.full((32, 32), float(i)) for i in range(4)]
        buf = tiles  # type: ignore  # Don't use Optional typing to avoid assignment error
        block = Block(buf, 4, Span(0, 4))  # type: ignore

        destination = tu.zeros(128, 32)  # 4 tiles

        tx = copy(block, destination)
        tx.wait()

        assert tx.is_completed
        # Verify each tile was placed correctly
        for i in range(4):
            start_row = i * TILE_SHAPE[0]
            end_row = (i + 1) * TILE_SHAPE[0]
            tile_slice = destination[start_row:end_row, :]
            assert tu.allclose(tile_slice, tiles[i])

    def test_transfer_shape_mismatch(self) -> None:
        """Test that shape mismatch between Block and tensor raises ValueError."""
        tile0 = tu.ones(32, 32)
        tile1 = tu.zeros(32, 32)
        buf: List[Optional[torch.Tensor]] = [tile0, tile1]
        block = Block(buf, 2, Span(0, 2))

        # Wrong destination shape
        destination = tu.zeros(96, 32)  # 3 tiles, but Block has 2

        with pytest.raises(ValueError, match="Expected 2 tiles but found 3"):
            tx = copy(block, destination)  # type: ignore

    def test_transfer_from_single_slot_block(self) -> None:
        """Test transferring from a Block with a single slot."""
        tile = tu.full((32, 32), 9.99)
        buf: List[Optional[torch.Tensor]] = [None]
        block = Block(buf, 1, Span(0, 1))
        block[0] = tile

        destination = tu.zeros(32, 32)

        tx = copy(block, destination)
        tx.wait()

        assert tx.is_completed
        assert tu.allclose(destination, tile)


class TestCopyConvenienceFunction:
    """Test the copy() convenience function."""

    def test_copy_function_creates_transaction(self) -> None:
        """Test that copy() function creates and returns a CopyTransaction."""
        source = tu.randn(64, 32)  # 2 tiles to match Block size
        buf: List[Optional[torch.Tensor]] = [None, None]
        block = Block(buf, 2, Span(0, 2))

        tx = copy(source, block)

        match tx:
            case CopyTransaction():
                assert not tx.is_completed
            case _:
                raise AssertionError(
                    f"Expected CopyTransaction, got {type(tx).__name__}"
                )


class TestCopyComplexOperations:
    """Test complex copy operation scenarios."""

    def test_multiple_sequential_transfers(self) -> None:
        """Test performing multiple copy transfers in sequence."""
        # Setup: Create source tensors
        tensor1 = tu.full((64, 32), 1.0)  # 2 tiles

        # Intermediate Block buffer
        buf: List[Optional[torch.Tensor]] = [None, None, None, None]
        block = Block(buf, 4, Span(0, 2))

        # Stage 1: Load first tensor to Block
        tx1 = copy(tensor1, block)
        tx1.wait()
        assert tx1.is_completed

        # Stage 2: Process tiles in Block (simulate some operation)
        processed_tiles = [block[i] * 10.0 for i in range(len(block))]
        block2 = Block(processed_tiles, 2, Span(0, 2))  # type: ignore

        # Stage 3: Extract processed data back to tensor
        result = tu.zeros(64, 32)
        tx2 = copy(block2, result)
        tx2.wait()
        assert tx2.is_completed

        # Verify the transformation
        expected = tensor1 * 10.0
        assert tu.allclose(result, expected)

    def test_copy_with_single_element_block(self) -> None:
        """Test copy with minimal Block (single element)."""
        source = tu.full((32, 32), 123.456)
        buf: List[Optional[torch.Tensor]] = [None]
        block = Block(buf, 1, Span(0, 1))

        # Transfer to Block
        tx1 = copy(source, block)
        tx1.wait()

        # Transfer back to tensor
        destination = tu.zeros(32, 32)
        tx2 = copy(block, destination)
        tx2.wait()

        assert tu.allclose(destination, source)


class TestCopyErrorHandling:
    """Test copy error conditions and edge cases."""

    def test_copy_with_empty_block(self) -> None:
        """Test copy behavior with zero-length Block."""
        source = tu.ones(32, 32)
        buf: List[Optional[torch.Tensor]] = []
        block = Block(buf, 0, Span(0, 0))

        # Should fail when trying to create copy to empty Block
        with pytest.raises(ValueError):
            tx = copy(source, block)  # type: ignore


class TestMulticastCopy:
    """Tests for pipe copy using the public `copy` API."""

    def test_pipe_single_tile_single_receiver(self) -> None:
        """Send a single tile via pipe and receive it."""
        tile = tu.full((32, 32), 123.0)
        src_buf: List[Optional[torch.Tensor]] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1))

        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1))

        pipe = Pipe(210, 211)

        tx_send = copy(src_block, pipe)
        tx_send.wait()

        tx_recv = copy(pipe, dst_block)
        tx_recv.wait()

        assert dst_block[0] is not None
        assert tu.allclose(dst_block[0], tile)

    def test_pipe_multiple_tiles_multiple_receivers(self) -> None:
        """Send multiple tiles and have multiple receivers consume them."""
        tile1 = tu.full((32, 32), 1.0)
        tile2 = tu.full((32, 32), 2.0)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2))

        # Cores 212 and 213 form a rectangular range in row 26: (26,4) to (26,5)
        pipe = Pipe((26, 3), ((26, 4), (26, 5)))

        tx_send = copy(src_block, pipe)
        tx_send.wait()

        # First receiver
        dst1: List[Optional[torch.Tensor]] = [None, None]
        dst_ring1 = Block(dst1, 2, Span(0, 2))
        tx_r1 = copy(pipe, dst_ring1)
        tx_r1.wait()
        assert tu.allclose(dst_ring1[0], tile1)
        assert tu.allclose(dst_ring1[1], tile2)

        # Second receiver
        dst2: List[Optional[torch.Tensor]] = [None, None]
        dst_ring2 = Block(dst2, 2, Span(0, 2))
        tx_r2 = copy(pipe, dst_ring2)
        tx_r2.wait()
        assert tu.allclose(dst_ring2[0], tile1)
        assert tu.allclose(dst_ring2[1], tile2)

    def test_pipe_length_mismatch_raises(self) -> None:
        """Receiver with mismatched length should raise ValueError at wait()."""
        tile1 = tu.ones(32, 32)
        tile2 = tu.zeros(32, 32)
        src_buf: List[Optional[torch.Tensor]] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2))

        # Single core unicast using 2D coordinates
        pipe = Pipe((26, 4), (26, 5))

        tx_send = copy(src_block, pipe)
        tx_send.wait()

        # Receiver with wrong length
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ring = Block(dst_buf, 1, Span(0, 1))
        tx_recv = copy(pipe, dst_ring)
        with pytest.raises(
            ValueError,
            match="Destination Block length .* does not match pipe data length",
        ):
            tx_recv.wait()

    def test_pipe_receive_timeout(self) -> None:
        """Receiving on an address with no send should timeout."""
        pipe = Pipe(99, 100)
        dst_buf: List[Optional[torch.Tensor]] = [None]
        dst_ring = Block(dst_buf, 1, Span(0, 1))

        tx_recv = copy(pipe, dst_ring)
        with pytest.raises(ValueError, match="Timeout waiting for pipe data"):
            tx_recv.wait()
