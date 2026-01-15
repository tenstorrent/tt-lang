# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for copy operation simulation.

Tests the copy transfer functionality between tensors and Blocks,
including error handling and edge cases.
"""

from typing import List

import pytest
from test_utils import (
    make_full_tile,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
    tensors_equal,
)

from python.sim.block import Block, Span
from python.sim.cbstate import CBSlot
from python.sim.copy import CopyTransaction, copy
from python.sim.typedefs import Pipe


class TestCopyTransaction:
    """Test CopyTransaction class functionality."""

    def test_copy_transaction_creation_tensor_to_block(self) -> None:
        """Test creating a copy transaction from tensor to Block."""
        tensor = make_rand_tensor(64, 32)  # 2x1 tile block
        buf: List[CBSlot] = [None, None]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        tx = CopyTransaction(tensor, block)
        assert not tx.is_completed

    def test_copy_transaction_creation_block_to_tensor(self) -> None:
        """Test creating a copy transaction from Block to tensor."""
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))
        tensor = make_rand_tensor(64, 32)  # 2 tiles to match block size

        tx = CopyTransaction(block, tensor)
        assert not tx.is_completed

    def test_copy_transaction_unsupported_types(self) -> None:
        """Test that unsupported type combinations raise ValueError."""
        tensor1 = make_rand_tensor(32, 32)
        tensor2 = make_zeros_tile()

        # tensor → tensor not supported
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Tensor, Tensor\\)"
        ):
            CopyTransaction(tensor1, tensor2)

        # Block → Block not supported
        buf: List[CBSlot] = [None, None]
        block1 = Block(buf, 2, Span(0, 2), shape=(2, 1))
        block2 = Block(buf, 2, Span(0, 2), shape=(2, 1))
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Block, Block\\)"
        ):
            CopyTransaction(block1, block2)


class TestTensorToBlockCopy:
    """Test copy operations from tensor to Block."""

    def test_transfer_single_tile_to_block(self) -> None:
        """Test transferring a single tile tensor to Block."""
        source = make_ones_tile()  # Single tile
        buf: List[CBSlot] = [None]  # Single slot for single tile
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        tx = copy(source, block)
        tx.wait()

        # Verify completion
        assert tx.is_completed

        # Verify data was transferred correctly
        assert block[0] is not None
        assert tensors_equal(block[0], source)

    def test_transfer_multi_tile_to_block(self) -> None:
        """Test transferring a multi-tile tensor to Block."""
        # Create 2x1 tile tensor (64x32)
        source = make_rand_tensor(64, 32)

        buf: List[CBSlot] = [None, None, None, None]
        block = Block(buf, 4, Span(0, 2), shape=(2, 1))

        tx = copy(source, block)
        tx.wait()

        assert tx.is_completed
        # Verify tiles were stored
        assert block[0] is not None
        assert block[1] is not None

    def test_transfer_mismatched_tile_count(self) -> None:
        """Test that mismatched tile count raises ValueError."""
        # 3 tiles in tensor but block expects 2 tiles
        source = make_rand_tensor(96, 32)  # 3x1 tiles
        buf: List[CBSlot] = [None, None, None]
        block = Block(buf, 3, Span(0, 2), shape=(2, 1))  # Expects 2x1 tiles

        with pytest.raises(
            ValueError, match="Tensor shape .* does not match Block shape"
        ):
            copy(source, block)

    def test_transfer_to_single_slot_block(self) -> None:
        """Test transferring to a Block with a single slot."""
        source = make_full_tile(42.0)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        tx = copy(source, block)
        tx.wait()

        assert tx.is_completed
        assert block[0] is not None
        assert tensors_equal(block[0], source)


class TestBlockToTensorCopy:
    """Test copy operations from Block to tensor."""

    def test_transfer_single_tile_from_block(self) -> None:
        """Test transferring a single tile from Block to tensor."""
        tile0 = make_full_tile(3.14)
        tile1 = make_full_tile(2.71)
        buf: List[CBSlot] = [tile0, tile1]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        destination = make_rand_tensor(64, 32)

        tx = copy(block, destination)
        tx.wait()

        assert tx.is_completed
        # Check that tiles were placed in destination
        dest_tile0 = destination[0:1, 0:1]
        dest_tile1 = destination[1:2, 0:1]
        assert tensors_equal(dest_tile0, tile0)
        assert tensors_equal(dest_tile1, tile1)

    def test_transfer_multi_tile_from_block(self) -> None:
        """Test transferring multiple tiles from Block to tensor."""
        tiles = [make_full_tile(float(i)) for i in range(4)]
        buf: List[CBSlot] = list(tiles)  # Cast to CBSlot list
        block = Block(buf, 4, Span(0, 4), shape=(4, 1))

        destination = make_rand_tensor(128, 32)  # 4 tiles

        tx = copy(block, destination)
        tx.wait()

        assert tx.is_completed
        # Verify each tile was placed correctly
        for i in range(4):
            dest_tile = destination[i : i + 1, 0:1]
            assert tensors_equal(dest_tile, tiles[i])

    def test_transfer_shape_mismatch(self) -> None:
        """Test that shape mismatch between Block and tensor raises ValueError."""
        tile0 = make_ones_tile()
        tile1 = make_zeros_tile()
        buf: List[CBSlot] = [tile0, tile1]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        # Wrong destination shape
        destination = make_rand_tensor(96, 32)  # 3x1 tiles, but Block is 2x1

        with pytest.raises(
            ValueError, match="Tensor shape .* does not match Block shape"
        ):
            copy(block, destination)

    def test_transfer_from_single_slot_block(self) -> None:
        """Test transferring from a Block with a single slot."""
        tile = make_full_tile(9.99)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))
        block[0] = tile

        destination = make_zeros_tile()

        tx = copy(block, destination)
        tx.wait()

        assert tx.is_completed
        assert tensors_equal(destination, tile)


class TestCopyConvenienceFunction:
    """Test the copy() convenience function."""

    def test_copy_function_creates_transaction(self) -> None:
        """Test that copy() function creates and returns a CopyTransaction."""
        source = make_rand_tensor(64, 32)  # 2 tiles to match Block size
        buf: List[CBSlot] = [None, None]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        tx = copy(source, block)

        # Verify it's a CopyTransaction
        assert isinstance(tx, CopyTransaction)
        assert not tx.is_completed


class TestCopyComplexOperations:
    """Test complex copy operation scenarios."""

    def test_multiple_sequential_transfers(self) -> None:
        """Test performing multiple copy transfers in sequence."""
        # Setup: Create source tensors
        source_2tiles = make_rand_tensor(64, 32)  # 2 tiles

        # Intermediate Block buffer
        buf: List[CBSlot] = [None, None, None, None]
        block = Block(buf, 4, Span(0, 2), shape=(2, 1))

        # Stage 1: Load first tensor to Block (as 2 tiles)
        tx1 = copy(source_2tiles, block)
        tx1.wait()
        assert tx1.is_completed

        # Stage 2: Process tiles in Block (simulate some operation)
        # Verify tiles exist
        assert block[0] is not None
        assert block[1] is not None
        processed_tiles = [block[0] * 10.0, block[1] * 10.0]
        processed_buf: List[CBSlot] = list(processed_tiles)
        block2 = Block(processed_buf, 2, Span(0, 2), shape=(2, 1))

        # Stage 3: Extract processed data back to tensor
        result = make_zeros_tile()
        # Copy first tile only for simplicity
        result_buf: List[CBSlot] = [None]
        result_block = Block(result_buf, 1, Span(0, 1), shape=(1, 1))
        result_block[0] = block2[0]
        tx2 = copy(result_block, result)
        tx2.wait()
        assert tx2.is_completed

    def test_copy_with_single_element_block(self) -> None:
        """Test copy with minimal Block (single element)."""
        source = make_full_tile(123.456)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        # Transfer to Block
        tx1 = copy(source, block)
        tx1.wait()

        # Transfer back to tensor
        destination = make_zeros_tile()
        tx2 = copy(block, destination)
        tx2.wait()

        assert tensors_equal(destination, source)


class TestCopyErrorHandling:
    """Test copy error conditions and edge cases."""

    def test_copy_with_empty_block(self) -> None:
        """Test copy behavior with zero-length Block."""
        source = make_ones_tile()
        buf: List[CBSlot] = []
        block = Block(buf, 0, Span(0, 0), shape=(0, 0))

        # Should fail when trying to create copy to empty Block
        with pytest.raises(ValueError):
            copy(source, block)


class TestMulticastCopy:
    """Tests for pipe copy using the public `copy` API."""

    def test_pipe_single_tile_single_receiver(self) -> None:
        """Send a single tile via pipe and receive it."""
        tile = make_full_tile(123.0)
        src_buf: List[CBSlot] = [tile]
        src_block = Block(src_buf, 1, Span(0, 1), shape=(1, 1))

        dst_buf: List[CBSlot] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1), shape=(1, 1))

        pipe = Pipe(210, 211)

        tx_send = copy(src_block, pipe)
        tx_send.wait()

        tx_recv = copy(pipe, dst_block)
        tx_recv.wait()

        assert dst_block[0] is not None
        assert tensors_equal(dst_block[0], tile)

    def test_pipe_multiple_tiles_multiple_receivers(self) -> None:
        """Send multiple tiles and have multiple receivers consume them."""
        tile1 = make_full_tile(1.0)
        tile2 = make_full_tile(2.0)
        src_buf: List[CBSlot] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2), shape=(2, 1))

        # Cores 212 and 213 form a rectangular range in row 26: (26,4) to (26,5)
        pipe = Pipe((26, 3), ((26, 4), (26, 5)))

        tx_send = copy(src_block, pipe)
        tx_send.wait()

        # First receiver
        dst1: List[CBSlot] = [None, None]
        dst_ring1 = Block(dst1, 2, Span(0, 2), shape=(2, 1))
        tx_r1 = copy(pipe, dst_ring1)
        tx_r1.wait()
        assert dst_ring1[0] is not None
        assert dst_ring1[1] is not None
        assert tensors_equal(dst_ring1[0], tile1)
        assert tensors_equal(dst_ring1[1], tile2)

        # Second receiver
        dst2: List[CBSlot] = [None, None]
        dst_ring2 = Block(dst2, 2, Span(0, 2), shape=(2, 1))
        tx_r2 = copy(pipe, dst_ring2)
        tx_r2.wait()
        assert dst_ring2[0] is not None
        assert dst_ring2[1] is not None
        assert tensors_equal(dst_ring2[0], tile1)
        assert tensors_equal(dst_ring2[1], tile2)

    def test_pipe_length_mismatch_raises(self) -> None:
        """Receiver with mismatched length should raise ValueError at wait()."""
        tile1 = make_ones_tile()
        tile2 = make_zeros_tile()
        src_buf: List[CBSlot] = [tile1, tile2]
        src_block = Block(src_buf, 2, Span(0, 2), shape=(2, 1))

        # Single core unicast using 2D coordinates
        pipe = Pipe((26, 4), (26, 5))

        tx_send = copy(src_block, pipe)
        tx_send.wait()

        # Receiver with wrong length
        dst_buf: List[CBSlot] = [None]
        dst_ring = Block(dst_buf, 1, Span(0, 1), shape=(1, 1))
        tx_recv = copy(pipe, dst_ring)
        with pytest.raises(
            ValueError,
            match="Destination Block length .* does not match pipe data length",
        ):
            tx_recv.wait()

    def test_pipe_receive_timeout(self) -> None:
        """Receiving on an address with no send should timeout."""
        pipe = Pipe(99, 100)
        dst_buf: List[CBSlot] = [None]
        dst_ring = Block(dst_buf, 1, Span(0, 1), shape=(1, 1))

        tx_recv = copy(pipe, dst_ring)
        with pytest.raises(TimeoutError, match="Timeout waiting for pipe data"):
            tx_recv.wait()


class TestCopyTransactionCanWait:
    """Test can_wait() functionality for CopyTransaction."""

    def test_can_wait_before_transfer(self) -> None:
        """Test that can_wait() returns True for synchronous Tensor→Block copies."""
        source = make_ones_tile()
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        tx = copy(source, block)
        # Tensor→Block is synchronous, can_wait() returns True immediately
        assert tx.can_wait() is True
        assert tx.is_completed is False

    def test_can_wait_after_transfer(self) -> None:
        """Test that can_wait() returns True after wait() completes."""
        source = make_ones_tile()
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        tx = copy(source, block)
        tx.wait()
        # After wait() completes, transfer is done
        assert tx.can_wait() is True
        assert tx.is_completed is True

    def test_can_wait_idempotent(self) -> None:
        """Test that can_wait() can be called multiple times."""
        source = make_full_tile(3.14)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        tx = copy(source, block)

        # Tensor→Block is synchronous, returns True before wait()
        assert tx.can_wait() is True
        assert tx.can_wait() is True  # Multiple calls

        tx.wait()

        # After wait, still True
        assert tx.can_wait() is True
        assert tx.can_wait() is True  # Multiple calls
        assert tx.can_wait() is True

    def test_can_wait_multiple_waits(self) -> None:
        """Test that wait() can be called multiple times after completion."""
        source = make_ones_tile()
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        tx = copy(source, block)
        # Tensor→Block is synchronous
        assert tx.can_wait() is True

        # First wait
        tx.wait()
        assert tx.can_wait() is True

        # Second wait (should be no-op since already completed)
        tx.wait()
        assert tx.can_wait() is True

        # Verify data is still correct
        assert block[0] is not None
        assert tensors_equal(block[0], source)

    def test_can_wait_with_multi_tile_transfer(self) -> None:
        """Test can_wait() with multi-tile transfer."""
        # Create 2x2 tile tensor
        source = make_rand_tensor(64, 64)  # 2x2 tiles
        buf: List[CBSlot] = [None, None, None, None]
        block = Block(buf, 4, Span(0, 4), shape=(2, 2))

        tx = copy(source, block)
        # Tensor→Block is synchronous
        assert tx.can_wait() is True
        assert not tx.is_completed

        tx.wait()
        assert tx.can_wait() is True
        assert tx.is_completed

    def test_can_wait_block_to_tensor(self) -> None:
        """Test can_wait() with Block to tensor transfer."""
        # Create source Block
        buf: List[CBSlot] = [make_ones_tile(), make_full_tile(2.0)]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        # Create destination tensor
        dst = make_rand_tensor(64, 32)  # 2x1 tiles

        tx = copy(block, dst)
        # Block→Tensor is synchronous
        assert tx.can_wait() is True

        tx.wait()
        assert tx.can_wait() is True
        assert tx.is_completed

    def test_can_wait_with_pipe(self) -> None:
        """Test can_wait() with pipe transfers."""
        # Setup pipe
        pipe = Pipe(10, 20)

        # Pipe→Block without data returns False
        dst_buf_empty: List[CBSlot] = [None]
        dst_block_empty = Block(dst_buf_empty, 1, Span(0, 1), shape=(1, 1))
        tx_recv_empty = copy(pipe, dst_block_empty)
        # Pipe has no data yet
        assert tx_recv_empty.can_wait() is False

        # Source (Block to Pipe) - synchronous
        src_buf: List[CBSlot] = [make_full_tile(5.0)]
        src_block = Block(src_buf, 1, Span(0, 1), shape=(1, 1))
        tx_send = copy(src_block, pipe)

        # Block→Pipe is synchronous
        assert tx_send.can_wait() is True
        tx_send.wait()
        assert tx_send.can_wait() is True

        # Now pipe has data, so can_wait() should return True
        assert tx_recv_empty.can_wait() is True

        # Destination (Pipe to Block) - asynchronous, data now available
        dst_buf: List[CBSlot] = [None]
        dst_block = Block(dst_buf, 1, Span(0, 1), shape=(1, 1))
        tx_recv = copy(pipe, dst_block)

        # Pipe→Block can proceed because pipe has data (from tx_send)
        assert tx_recv.can_wait() is True
        tx_recv.wait()
        # After consuming the data, pipe is empty again, so can_wait() returns False
        assert tx_recv.can_wait() is False

        # Verify data
        assert dst_block[0] is not None
        assert src_block[0] is not None
        assert tensors_equal(dst_block[0], src_block[0])
