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
from python.sim.cbapi import CBAPI
from python.sim.cbstate import CBSlot
from python.sim.copy import CopyTransaction, copy
from python.sim.typedefs import Pipe


@pytest.fixture
def api():
    """Provide a fresh CBAPI instance for each test."""
    return CBAPI()


class TestCopyTransaction:
    """Test CopyTransaction class functionality."""

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


class TestBlockToTensorCopy:
    """Test copy operations from Block to tensor."""

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


class TestCopyConvenienceFunction:
    """Test the copy() convenience function."""

    # All tests removed - covered by TestCopyWithStateMachine


class TestCopyComplexOperations:
    """Test complex copy operation scenarios."""

    # All tests removed - covered by TestCopyWithStateMachine


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

    # All tests removed - covered by TestCopyWithStateMachine


class TestCopyTransactionCanWait:
    """Test can_wait() functionality for CopyTransaction."""

    # All tests removed - covered by TestCopyWithStateMachine


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
            match="Cannot write to Block.*has no access.*NAR state",
        ):
            source_block.store([make_zeros_tile(), make_zeros_tile()])

        # After wait(), the block still doesn't support store() because it's a wait() block
        tx.wait()
        # wait() blocks cannot use store() per state machine - they expect STORE_SRC
        with pytest.raises(RuntimeError, match="Impossible.*Invalid state for store"):
            source_block.store(
                [make_zeros_tile(), make_zeros_tile()]
            )  # Should still fail

    # Removed: test_can_read_from_block_source_before_wait - covered by TestCopyWithStateMachine


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

        # Attempt to read from destination should fail (dest is in NAW state)
        with pytest.raises(
            RuntimeError,
            match="Cannot read from Block.*NAW state",
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

        # Attempt to write to destination should fail (dest is in NAW state)
        with pytest.raises(
            RuntimeError,
            match="Cannot write to Block.*copy destination.*copy lock error.*NAW",
        ):
            dest_block.store([make_ones_tile(), make_ones_tile()])

        # After wait(), block expects PUSH (not store) per state machine
        tx.wait()
        # Cannot store on DM block - only Compute blocks support store
        with pytest.raises(
            RuntimeError,
            match="Impossible.*Invalid state for store",
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

    # Removed: test_can_read_source_multiple_times - tests multiple copies which is not allowed per state machine


class TestCopyLockingAfterWait:
    """Test that locks are released after wait() completes."""

    # All tests removed - covered by TestCopyWithStateMachine


class TestCopyWaitIdempotency:
    """Test that calling wait() multiple times is safe."""

    # All tests removed - covered by TestCopyWithStateMachine


class TestCopyWithStateMachine:
    """Test copy operations using CircularBuffer (conforming to state machine)."""

    def test_copy_tensor_to_block_with_reserve(self, api: "CBAPI") -> None:
        """Test Tensor -> Block copy using reserve() in DM thread."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        # Set DM thread context for copy operations
        _set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 32)  # 2x1 tiles
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()
            # Verify data transferred correctly
            assert block[0] is not None
            assert block[1] is not None

    def test_copy_block_to_tensor_with_wait(self, api: "CBAPI") -> None:
        """Test Block -> Tensor copy using wait() in DM thread."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        # Setup: Fill CB with data using reserve->store->push pattern
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )
        source = make_rand_tensor(64, 32)

        with cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Now copy from CB to tensor
        destination = make_rand_tensor(64, 32)
        with cb.wait() as block:
            tx = copy(block, destination)
            tx.wait()

        # Verify tiles in destination match source
        dest_tile0 = destination[0:1, 0:1]
        dest_tile1 = destination[1:2, 0:1]
        source_tile0 = source[0:1, 0:1]
        source_tile1 = source[1:2, 0:1]
        assert tensors_equal(dest_tile0, source_tile0)
        assert tensors_equal(dest_tile1, source_tile1)

    def test_copy_single_tile_tensor_to_block(self, api: "CBAPI") -> None:
        """Test single tile Tensor -> Block copy."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()
            assert block[0] is not None
            assert tensors_equal(block[0], source)

    def test_copy_multi_tile_tensor_to_block(self, api: "CBAPI") -> None:
        """Test multi-tile Tensor -> Block copy."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(128, 32)  # 4x1 tiles
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(4, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()
            # Verify all tiles transferred
            for i in range(4):
                assert block[i] is not None

    def test_copy_with_pipe_single_tile(self, api: "CBAPI") -> None:
        """Test Block -> Pipe -> Block copy with single tile."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        tile = make_full_tile(123.0)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        pipe = Pipe(210, 211)

        # Send tile to src_cb
        with src_cb.reserve() as block:
            tx = copy(tile, block)
            tx.wait()

        # Copy from src_cb to pipe, then immediately copy from pipe to dst_cb
        with src_cb.wait() as src_block:
            with dst_cb.reserve() as dst_block:
                tx_send = copy(src_block, pipe)
                tx_send.wait()
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()

        # Verify data in destination by reading (won't pop, just read)
        result = make_zeros_tile()
        with dst_cb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_copy_with_pipe_multiple_tiles(self, api: "CBAPI") -> None:
        """Test Block -> Pipe -> Block copy with multiple tiles."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)
        grid = (100, 100)  # Set grid context for pipe operations

        source = make_rand_tensor(64, 32)  # 2x1 tiles
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )
        pipe = Pipe((26, 3), (26, slice(4, 6)))

        # Fill source CB
        with src_cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Copy from src_cb to pipe, then immediately copy from pipe to dst_cb
        with src_cb.wait() as src_block:
            with dst_cb.reserve() as dst_block:
                tx_send = copy(src_block, pipe)
                tx_send.wait()
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()

        # Verify data in destination
        result = make_rand_tensor(64, 32)
        with dst_cb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        # Verify tiles match source
        result_tile0 = result[0:1, 0:1]
        result_tile1 = result[1:2, 0:1]
        source_tile0 = source[0:1, 0:1]
        source_tile1 = source[1:2, 0:1]
        assert tensors_equal(result_tile0, source_tile0)
        assert tensors_equal(result_tile1, source_tile1)

    def test_copy_sequential_transfers(self, api: "CBAPI") -> None:
        """Test multiple sequential copy operations."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 32)  # 2 tiles
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )
        result = make_rand_tensor(64, 32)

        # Stage 1: Load tensor to CB
        with cb.reserve() as block:
            tx1 = copy(source, block)
            tx1.wait()
            assert block[0] is not None
            assert block[1] is not None

        # Stage 2: Extract from CB to result tensor
        with cb.wait() as block:
            tx2 = copy(block, result)
            tx2.wait()

        # Verify data in result matches source
        result_tile0 = result[0:1, 0:1]
        result_tile1 = result[1:2, 0:1]
        source_tile0 = source[0:1, 0:1]
        source_tile1 = source[1:2, 0:1]
        assert tensors_equal(result_tile0, source_tile0)
        assert tensors_equal(result_tile1, source_tile1)

    def test_copy_wait_idempotency(self, api: "CBAPI") -> None:
        """Test that calling wait() multiple times is safe."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            # Call wait multiple times
            tx.wait()
            tx.wait()
            tx.wait()
            # Should still work
            assert block[0] is not None

    def test_copy_can_wait_before_and_after(self, api: "CBAPI") -> None:
        """Test can_wait() functionality."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            # Tensor->Block is synchronous, can_wait() returns True immediately
            assert tx.can_wait() is True
            assert tx.is_completed is False

            tx.wait()
            # After wait, still True
            assert tx.can_wait() is True
            assert tx.is_completed is True

    def test_copy_multi_tile_can_wait(self, api: "CBAPI") -> None:
        """Test can_wait() with multi-tile transfer."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 64)  # 2x2 tiles
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 2), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            assert tx.can_wait() is True
            assert not tx.is_completed

            tx.wait()
            assert tx.can_wait() is True
            assert tx.is_completed

    def test_copy_with_pipe_can_wait(self, api: "CBAPI") -> None:
        """Test can_wait() with pipe transfers."""
        from python.sim.block import _set_current_thread_type
        from python.sim.cb import CircularBuffer

        _set_current_thread_type(ThreadType.DM)

        pipe = Pipe(10, 20)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Send data to pipe
        tile = make_full_tile(5.0)
        with src_cb.reserve() as src_block:
            tx_setup = copy(tile, src_block)
            tx_setup.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            # Block->Pipe is synchronous
            assert tx_send.can_wait() is True
            tx_send.wait()
            assert tx_send.can_wait() is True

        # Now receive from pipe (has data)
        with dst_cb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            assert tx_recv.can_wait() is True
            tx_recv.wait()
            # After consuming, pipe is empty
            assert tx_recv.can_wait() is False


class TestCopyTransactionProperties:
    """Test CopyTransaction properties and state."""

    def test_is_completed_property(self, api: "CBAPI") -> None:
        """Test that is_completed property correctly reflects transaction state."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)

            # Before wait(), transaction is not completed
            assert tx.is_completed is False

            tx.wait()

            # After wait(), transaction is completed
            assert tx.is_completed is True

            # Multiple property accesses should work
            assert tx.is_completed is True
            assert tx.is_completed is True

    def test_multiple_wait_on_completed_transaction(self, api: "CBAPI") -> None:
        """Test that calling wait() multiple times on completed transaction is safe."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 32)
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)

            # First wait completes the transaction
            tx.wait()
            assert tx.is_completed is True

            # Subsequent waits should be no-ops
            tx.wait()
            assert tx.is_completed is True
            tx.wait()
            assert tx.is_completed is True

            # Data should still be correct
            assert block[0] is not None
            assert block[1] is not None

    def test_can_wait_reflects_handler_behavior(self, api: "CBAPI") -> None:
        """Test that can_wait() correctly delegates to handler."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        # Tensor -> Block is always synchronous
        source = make_ones_tile()
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            assert tx.can_wait() is True  # Synchronous transfer
            assert tx.is_completed is False  # But not completed until wait()

            tx.wait()
            assert tx.can_wait() is True  # Still can call wait()
            assert tx.is_completed is True  # Now completed


class TestCopyContextManagerExtraction:
    """Test that copy works with both raw blocks and context managers."""

    def test_copy_with_context_managers(self, api: "CBAPI") -> None:
        """Test copy operations using context managers with Pipe."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        source = make_full_tile(42.0)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        pipe = Pipe(1000, 1001)

        # Use context managers directly in copy calls
        with src_cb.reserve() as src_ctx:
            # Pass context manager to copy
            tx = copy(source, src_ctx)
            tx.wait()

        # Copy through pipe using context managers
        with src_cb.wait() as src_ctx:
            # WaitContext -> Pipe
            tx = copy(src_ctx, pipe)
            tx.wait()

        with dst_cb.reserve() as dst_ctx:
            # Pipe -> ReserveContext
            tx = copy(pipe, dst_ctx)
            tx.wait()

        # Verify data was transferred
        result = make_zeros_tile()
        with dst_cb.wait() as dst_ctx:
            tx = copy(dst_ctx, result)
            tx.wait()

        assert tensors_equal(result, source)

    def test_mixed_context_managers_and_tensors(self, api: "CBAPI") -> None:
        """Test mixing context managers with raw tensors."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        source = make_full_tile(3.14)
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Tensor -> Context manager
        with cb.reserve() as ctx:
            tx = copy(source, ctx)
            tx.wait()

        # Context manager -> Tensor
        result = make_zeros_tile()
        with cb.wait() as ctx:
            tx = copy(ctx, result)
            tx.wait()

        assert tensors_equal(result, source)


class TestCopyErrorConditions:
    """Test error conditions and edge cases in copy operations."""

    def test_copy_creates_transaction_immediately(self, api: "CBAPI") -> None:
        """Test that copy() creates transaction immediately, not on wait()."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy, CopyTransaction

        _set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            # copy() should return a CopyTransaction immediately
            tx = copy(source, block)
            assert isinstance(tx, CopyTransaction)
            assert tx.is_completed is False

            # Transaction exists before wait()
            assert tx.can_wait() is True

            tx.wait()
            assert tx.is_completed is True

    def test_unsupported_type_combinations_raise_valueerror(self) -> None:
        """Test that unsupported copy type combinations raise ValueError."""
        from python.sim.copy import copy

        tensor1 = make_ones_tile()
        tensor2 = make_zeros_tile()

        # Tensor -> Tensor is not supported
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Tensor, Tensor\\)"
        ):
            copy(tensor1, tensor2)


if __name__ == "__main__":
    pytest.main([__file__])
