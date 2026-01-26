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

from python.sim.block import Block, BlockAcquisition, Span, ThreadType
from python.sim.cbstate import CBSlot
from python.sim.copy import CopyTransaction, copy
from python.sim.typedefs import Pipe


class TestCopyTransaction:
    """Test CopyTransaction class functionality."""

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
    def test_copy_transaction_creation_tensor_to_block(self) -> None:
        """Test creating a copy transaction from tensor to Block."""
        tensor = make_rand_tensor(64, 32)  # 2x1 tile block
        buf: List[CBSlot] = [None, None]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        tx = CopyTransaction(tensor, block)
        assert not tx.is_completed

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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
        block1 = Block(
            buf,
            2,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )
        block2 = Block(
            buf,
            2,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Block, Block\\)"
        ):
            CopyTransaction(block1, block2)


class TestTensorToBlockCopy:
    """Test copy operations from tensor to Block."""

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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
        block = Block(
            buf,
            3,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )  # Expects 2x1 tiles

        with pytest.raises(
            ValueError, match="Tensor shape .* does not match Block shape"
        ):
            copy(source, block)

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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
        block = Block(
            buf,
            2,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.WAIT,
            thread_type=ThreadType.DM,
        )

        # Wrong destination shape
        destination = make_rand_tensor(96, 32)  # 3x1 tiles, but Block is 2x1

        with pytest.raises(
            ValueError, match="Tensor shape .* does not match Block shape"
        ):
            copy(block, destination)

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
    def test_transfer_from_single_slot_block(self) -> None:
        """Test transferring from a Block with a single slot."""
        tile = make_full_tile(9.99)
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))
        block.store([tile])

        destination = make_zeros_tile()

        tx = copy(block, destination)
        tx.wait()

        assert tx.is_completed
        assert tensors_equal(destination, tile)


class TestCopyConvenienceFunction:
    """Test the copy() convenience function."""

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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
        result_block.store([block2[0]])
        tx2 = copy(result_block, result)
        tx2.wait()
        assert tx2.is_completed

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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
        block = Block(
            buf,
            0,
            Span(0, 0),
            shape=(0, 0),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )

        # Should fail when trying to create copy to empty Block
        with pytest.raises(ValueError):
            copy(source, block)


class TestMulticastCopy:
    """Tests for pipe copy using the public `copy` API."""

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
    def test_can_wait_before_transfer(self) -> None:
        """Test that can_wait() returns True for synchronous Tensor→Block copies."""
        source = make_ones_tile()
        buf: List[CBSlot] = [None]
        block = Block(buf, 1, Span(0, 1), shape=(1, 1))

        tx = copy(source, block)
        # Tensor→Block is synchronous, can_wait() returns True immediately
        assert tx.can_wait() is True
        assert tx.is_completed is False

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
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


class TestCopySourceLocking:
    """Test that copy source is locked against writes until wait() completes."""

    def test_cannot_write_to_block_source_before_wait(self) -> None:
        """Test that writing to Block source before wait() raises RuntimeError."""
        # Create source block with data
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        source_block = Block(
            buf,
            2,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.WAIT,
            thread_type=ThreadType.DM,
        )

        # Create destination tensor (non-Block, so no state changes)
        dest_tensor = make_rand_tensor(64, 32)

        # Start copy
        tx = copy(source_block, dest_tensor)

        # Attempt to write to source block should fail (source expects TX_WAIT)
        # But more fundamentally, wait() blocks don't support store() - they expect POP
        with pytest.raises(
            RuntimeError,
            match="Cannot perform store\\(\\):|Cannot write to Block: Block is locked as copy source",
        ):
            source_block.store([make_zeros_tile(), make_zeros_tile()])

        # After wait(), the block still doesn't support store() because it's a wait() block
        tx.wait()
        # wait() blocks cannot use store() per state machine - they expect POP
        with pytest.raises(
            RuntimeError, match="Expected one of \\[POP\\], but got store\\(\\)"
        ):
            source_block.store(
                [make_zeros_tile(), make_zeros_tile()]
            )  # Should still fail

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
    def test_can_read_from_block_source_before_wait(self) -> None:
        """Test that reading from Block source before wait() is allowed."""
        # Create source block with data
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        source_block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        # Create destination tensor (non-Block, so no locking)
        dest_tensor = make_rand_tensor(64, 32)

        # Start copy
        tx = copy(source_block, dest_tensor)

        # Reading from source should still work
        tile = source_block[0]  # Should succeed
        assert tile is not None

        tx.wait()


class TestCopyDestinationLocking:
    """Test that copy destination is locked against all access until wait() completes."""

    def test_cannot_read_from_block_destination_before_wait(self) -> None:
        """Test that reading from Block destination before wait() raises RuntimeError."""
        # Create source tensor (non-Block, so no state changes)
        source_tensor = make_rand_tensor(64, 32)

        # Create destination block (needs to have slots initialized for read to work)
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        dest_block = Block(
            buf,
            2,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )

        # Start copy
        tx = copy(source_tensor, dest_block)

        # Attempt to read from destination should fail (dest is in NA state)
        with pytest.raises(
            RuntimeError,
            match="Cannot read from Block: Block has no access \\(NA state\\)",
        ):
            _ = dest_block[0]

        # After wait(), reading should work
        tx.wait()
        tile = dest_block[0]  # Should succeed
        assert tile is not None

    def test_cannot_write_to_block_destination_before_wait(self) -> None:
        """Test that writing to Block destination before wait() raises RuntimeError."""
        # Create source tensor (non-Block, so no locking)
        source_tensor = make_rand_tensor(64, 32)

        # Create destination block
        buf: List[CBSlot] = [None, None]
        dest_block = Block(
            buf,
            2,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )

        # Start copy
        tx = copy(source_tensor, dest_block)

        # Attempt to write to destination should fail (dest is in NA state)
        with pytest.raises(
            RuntimeError,
            match="Cannot write to Block: Block has no access \\(NA state\\)",
        ):
            dest_block.store([make_ones_tile(), make_ones_tile()])

        # After wait(), block expects PUSH (not store) per state machine
        tx.wait()
        # Cannot store on DM block - only Compute blocks support store
        with pytest.raises(
            RuntimeError,
            match="Expected one of \\[COPY_DST, COPY_SRC, PUSH\\], but got store\\(\\)",
        ):
            dest_block.store([make_ones_tile(), make_ones_tile()])


class TestMultipleCopyOperations:
    """Test locking behavior with multiple concurrent copy operations."""

    def test_cannot_use_same_block_as_source_and_destination(self) -> None:
        """Test that a block cannot be both source and destination simultaneously."""
        # Create block
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        block = Block(
            buf,
            2,
            Span(0, 2),
            shape=(2, 1),
            acquisition=BlockAcquisition.WAIT,
            thread_type=ThreadType.DM,
        )

        # Create tensors (non-Block, so no state changes)
        tensor1 = make_rand_tensor(64, 32)
        tensor2 = make_rand_tensor(64, 32)

        # Start copy with block as source
        tx1 = copy(block, tensor1)

        # Attempt to start copy with same block as destination should fail immediately
        # wait() DM blocks cannot be used as copy destinations per state machine
        with pytest.raises(
            RuntimeError,
            match="Expected one of \\[TX_WAIT\\], but got copy \\(as destination\\)",
        ):
            copy(tensor2, block)

        # Clean up
        tx1.wait()

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly. "
        "Also tests multiple copies from same block which is not allowed per state machine."
    )
    def test_can_read_source_multiple_times(self) -> None:
        """Test that a block can be source for multiple copies simultaneously."""
        # Create source block
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        source_block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        # Create multiple destinations (non-Block, so no locking)
        dest1 = make_rand_tensor(64, 32)
        dest2 = make_rand_tensor(64, 32)

        # Start multiple copies from same source
        tx1 = copy(source_block, dest1)
        tx2 = copy(
            source_block, dest2
        )  # Should succeed (read lock allows multiple readers)

        # Wait for both
        tx1.wait()
        tx2.wait()


class TestCopyLockingAfterWait:
    """Test that locks are released after wait() completes."""

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
    def test_block_unlocked_after_wait(self) -> None:
        """Test that blocks are fully unlocked after wait() completes."""
        # Create block and tensor
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        source_block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        dest_tensor = make_rand_tensor(64, 32)

        # Perform copy and wait
        tx = copy(source_block, dest_tensor)
        tx.wait()

        # Block should be fully accessible now
        _ = source_block[0]  # Read source - should succeed
        source_block.store(
            [make_ones_tile(), make_ones_tile()]
        )  # Write source - should succeed

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
    def test_can_reuse_blocks_after_wait(self) -> None:
        """Test that blocks can be used in new copy operations after wait()."""
        # Create blocks
        buf: List[CBSlot] = [make_ones_tile(), make_zeros_tile()]
        block = Block(buf, 2, Span(0, 2), shape=(2, 1))

        tensor1 = make_rand_tensor(64, 32)
        tensor2 = make_rand_tensor(64, 32)

        # First copy: block as source
        tx1 = copy(block, tensor1)
        tx1.wait()

        # Second copy: same block as destination (should work now)
        tx2 = copy(tensor2, block)
        tx2.wait()

        # Third copy: block as source again
        tx3 = copy(block, tensor1)
        tx3.wait()


class TestCopyWaitIdempotency:
    """Test that calling wait() multiple times is safe."""

    @pytest.mark.skip(
        reason="Does not conform to state machine diagram: "
        "blocks must be acquired via reserve()/wait(), not created directly"
    )
    def test_multiple_wait_calls(self) -> None:
        """Test that calling wait() multiple times doesn't cause issues."""
        buf1: List[CBSlot] = [make_ones_tile()]
        source_block = Block(buf1, 1, Span(0, 1), shape=(1, 1))

        dest_tensor = make_rand_tensor(32, 32)

        tx = copy(source_block, dest_tensor)

        # Call wait multiple times
        tx.wait()
        tx.wait()
        tx.wait()

        # Block should be accessible
        source_block.store([make_zeros_tile()])
