# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for copy operation simulation.

Tests the copy transfer functionality between tensors and Blocks,
including error handling and edge cases.
"""

import pytest
from test_utils import (
    make_element_for_buffer_shape,
    make_full_tile,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
    tensors_equal,
)

from python.sim.blockstate import BlockAcquisition, ThreadType
from python.sim.context import set_current_thread_type
from python.sim.dfb import Block, DataflowBuffer
from python.sim.ttnnsim import Tensor
from python.sim.copy import CopyTransaction, copy
from python.sim.pipe import Pipe


@pytest.fixture(autouse=True)
def setup_scheduler_context(dm_thread_context):
    """Automatically set scheduler context for all copy tests.

    Copy operations typically happen in DM threads.
    """
    # Use the shared dm_thread_context fixture
    pass


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
        block1 = Block(
            make_rand_tensor(64, 32),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )
        block2 = Block(
            make_rand_tensor(64, 32),
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
        """Test that mismatched element shape raises ValueError."""
        # Tensor is 3x1 tiles (96x32) but block expects 2x1 tiles (64x32)
        source = make_rand_tensor(96, 32)
        block = Block(
            make_rand_tensor(64, 32),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )

        with pytest.raises(ValueError, match="does not match Block shape"):
            copy(source, block)


class TestBlockToTensorCopy:
    """Test copy operations from Block to tensor."""

    def test_transfer_shape_mismatch(self) -> None:
        """Test that shape mismatch between Block and tensor raises ValueError."""
        block = Block(
            make_rand_tensor(64, 32),
            shape=(2, 1),
            acquisition=BlockAcquisition.WAIT,
            thread_type=ThreadType.DM,
        )

        # Wrong destination shape
        destination = make_rand_tensor(96, 32)  # 3x1 tiles, but Block is 2x1

        with pytest.raises(ValueError, match="does not match Tensor shape"):
            copy(block, destination)


class TestCopyConvenienceFunction:
    """Test the copy() convenience function."""

    # All tests removed - covered by TestCopyWithStateMachine


class TestCopyComplexOperations:
    """Test complex copy operation scenarios."""

    # All tests removed - covered by TestCopyWithStateMachine


class TestCopyErrorHandling:
    """Test copy error conditions and edge cases."""

    pass  # Remaining error cases are covered by TestCopyWithStateMachine


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
        source_block = Block(
            make_rand_tensor(64, 32),
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
            match="Cannot write to Block.*has no access.*ROR state",
        ):
            source_block.store(Block.from_tensor(make_rand_tensor(64, 32)))

        # After wait(), the block still doesn't support store() because it's a wait() block
        tx.wait()
        # wait() blocks cannot use store() per state machine - they expect STORE_SRC
        with pytest.raises(RuntimeError, match="Cannot perform store.*Expected one of"):
            source_block.store(Block.from_tensor(make_rand_tensor(64, 32)))

    # Removed: test_can_read_from_block_source_before_wait - covered by TestCopyWithStateMachine


class TestCopyDestinationLocking:
    """Test that copy destination is locked against all access until wait() completes."""

    def test_cannot_read_from_block_destination_before_wait(self) -> None:
        """Test that reading from Block destination before wait() raises RuntimeError."""
        # Create source tensor (non-Block, so no state changes)
        source_tensor = make_rand_tensor(64, 32)

        # Create destination block
        dest_block = Block(
            make_rand_tensor(64, 32),
            shape=(2, 1),
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.DM,
        )

        # Start copy
        tx = copy(source_tensor, dest_block)

        # Attempt to read from destination should fail (block indexing not allowed)
        with pytest.raises(
            RuntimeError,
            match="Block indexing.*not allowed",
        ):
            _ = dest_block[0]

        # After wait(), block indexing still not allowed (by design)
        tx.wait()

    def test_cannot_write_to_block_destination_before_wait(self) -> None:
        """Test that writing to Block destination before wait() raises RuntimeError."""
        # Create source tensor (non-Block, so no locking)
        source_tensor = make_rand_tensor(64, 32)

        # Create destination block
        dest_block = Block(
            make_rand_tensor(64, 32),
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
            dest_block.store(Block.from_tensor(make_rand_tensor(64, 32)))

        # After wait(), block expects PUSH (not store) per state machine
        tx.wait()
        # Cannot store on DM block - only Compute blocks support store
        with pytest.raises(
            RuntimeError,
            match="Cannot perform store.*Expected one of",
        ):
            dest_block.store(Block.from_tensor(make_rand_tensor(64, 32)))


class TestMultipleCopyOperations:
    """Test locking behavior with multiple concurrent copy operations."""

    def test_cannot_use_same_block_as_source_and_destination(self) -> None:
        """Test that a block cannot be both source and destination simultaneously."""
        # Create block
        block = Block(
            make_rand_tensor(64, 32),
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
            match="Expected one of \\[COPY_SRC, TX_WAIT\\], but got copy \\(as destination\\)",
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
    """Test copy operations using DataflowBuffer (conforming to state machine)."""

    def test_copy_tensor_to_block_with_reserve(self) -> None:
        """Test Tensor -> Block copy using reserve() in DM thread."""

        # Set DM thread context for copy operations
        set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 32)  # 2x1 tiles
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

            # Verify data was copied correctly
            block_data = block.to_list()
            assert tensors_equal(block_data[0], source[0:1, 0:1])

    def test_copy_block_to_tensor_with_wait(self) -> None:
        """Test Block -> Tensor copy using wait() in DM thread."""

        set_current_thread_type(ThreadType.DM)

        # Setup: Fill DFB with data using reserve->store->push pattern
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=2,
        )
        source = make_rand_tensor(64, 32)

        with dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Now copy from DFB to tensor
        destination = make_rand_tensor(64, 32)
        with dfb.wait() as block:
            tx = copy(block, destination)
            tx.wait()

        # Verify tiles in destination match source
        dest_tile0 = destination[0:1, 0:1]
        dest_tile1 = destination[1:2, 0:1]
        source_tile0 = source[0:1, 0:1]
        source_tile1 = source[1:2, 0:1]
        assert tensors_equal(dest_tile0, source_tile0)
        assert tensors_equal(dest_tile1, source_tile1)

    def test_copy_single_tile_tensor_to_block(self) -> None:
        """Test single tile Tensor -> Block copy."""

        set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

            # Verify data in block matches source
            block_data = block.to_list()
            assert tensors_equal(block_data[0], source)

    def test_copy_multi_tile_tensor_to_block(self) -> None:
        """Test multi-tile Tensor -> Block copy."""

        set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(128, 32)  # 4x1 tiles
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((4, 1)),
            shape=(4, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

            # Verify data in block matches source tiles
            block_data = block.to_list()
            for i in range(4):
                assert tensors_equal(block_data[i], source[i : i + 1, 0:1])

    def test_copy_with_pipe_single_tile(self) -> None:
        """Test Block -> Pipe -> Block copy with single tile."""

        set_current_thread_type(ThreadType.DM)

        tile = make_full_tile(123.0)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
        )
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
        )
        pipe = Pipe(210, 211)

        # Send tile to src_dfb
        with src_dfb.reserve() as block:
            tx = copy(tile, block)
            tx.wait()

        # Copy from src_dfb to pipe, then immediately copy from pipe to dst_dfb
        with src_dfb.wait() as src_block:
            with dst_dfb.reserve() as dst_block:
                tx_send = copy(src_block, pipe)
                tx_send.wait()
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()

        # Verify data in destination by reading (won't pop, just read)
        result = make_zeros_tile()
        with dst_dfb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_copy_with_pipe_multiple_tiles(self) -> None:
        """Test Block -> Pipe -> Block copy with multiple tiles."""

        set_current_thread_type(ThreadType.DM)
        grid = (100, 100)  # Set grid context for pipe operations

        source = make_rand_tensor(64, 32)  # 2x1 tiles
        src_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=2,
        )
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=2,
        )
        pipe = Pipe((26, 3), (26, slice(4, 6)))

        # Fill source DFB
        with src_dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Copy from src_dfb to pipe, then immediately copy from pipe to dst_dfb
        with src_dfb.wait() as src_block:
            with dst_dfb.reserve() as dst_block:
                tx_send = copy(src_block, pipe)
                tx_send.wait()
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()

        # Verify data in destination
        result = make_rand_tensor(64, 32)
        with dst_dfb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        # Verify tiles match source
        result_tile0 = result[0:1, 0:1]
        result_tile1 = result[1:2, 0:1]
        source_tile0 = source[0:1, 0:1]
        source_tile1 = source[1:2, 0:1]
        assert tensors_equal(result_tile0, source_tile0)
        assert tensors_equal(result_tile1, source_tile1)

    def test_copy_sequential_transfers(self) -> None:
        """Test multiple sequential copy operations."""

        set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 32)  # 2 tiles
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=2,
        )
        result = make_rand_tensor(64, 32)

        # Stage 1: Load tensor to DFB
        with dfb.reserve() as block:
            tx1 = copy(source, block)
            tx1.wait()

        # Stage 2: Extract from DFB to result tensor
        with dfb.wait() as block:
            tx2 = copy(block, result)
            tx2.wait()

        # Verify data in result matches source
        result_tile0 = result[0:1, 0:1]
        result_tile1 = result[1:2, 0:1]
        source_tile0 = source[0:1, 0:1]
        source_tile1 = source[1:2, 0:1]
        assert tensors_equal(result_tile0, source_tile0)
        assert tensors_equal(result_tile1, source_tile1)

    def test_copy_wait_idempotency(self) -> None:
        """Test that calling wait() multiple times is safe."""

        set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            # Call wait multiple times
            tx.wait()
            tx.wait()
            tx.wait()

            # Verify data was copied correctly
            block_data = block.to_list()
            assert tensors_equal(block_data[0], source)

    def test_copy_can_wait_before_and_after(self) -> None:
        """Test can_wait() functionality."""

        set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            # Tensor->Block is synchronous, can_wait() returns True immediately
            assert tx.can_wait() is True
            assert tx.is_completed is False

            tx.wait()
            # After wait, still True
            assert tx.can_wait() is True
            assert tx.is_completed is True

    def test_copy_multi_tile_can_wait(self) -> None:
        """Test can_wait() with multi-tile transfer."""

        set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 64)  # 2x2 tiles
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 2)),
            shape=(2, 2),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            assert tx.can_wait() is True
            assert not tx.is_completed

            tx.wait()
            assert tx.can_wait() is True
            assert tx.is_completed

    def test_copy_with_pipe_can_wait(self) -> None:
        """Test can_wait() with pipe transfers."""

        set_current_thread_type(ThreadType.DM)

        pipe = Pipe(10, 20)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
        )
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
        )

        # Send data to pipe
        tile = make_full_tile(5.0)
        with src_dfb.reserve() as src_block:
            tx_setup = copy(tile, src_block)
            tx_setup.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            # Block->Pipe is synchronous
            assert tx_send.can_wait() is True
            tx_send.wait()
            assert tx_send.can_wait() is True

        # Now receive from pipe (has data)
        with dst_dfb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            assert tx_recv.can_wait() is True
            tx_recv.wait()
            # After consuming, pipe is empty
            assert tx_recv.can_wait() is False


class TestCopyTransactionProperties:
    """Test CopyTransaction properties and state."""

    def test_is_completed_property(self) -> None:
        """Test that is_completed property correctly reflects transaction state."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)

            # Before wait(), transaction is not completed
            assert tx.is_completed is False

            tx.wait()

            # After wait(), transaction is completed
            assert tx.is_completed is True

            # Multiple property accesses should work
            assert tx.is_completed is True
            assert tx.is_completed is True

    def test_multiple_wait_on_completed_transaction(self) -> None:
        """Test that calling wait() multiple times on completed transaction is safe."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_rand_tensor(64, 32)
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)

            # First wait completes the transaction
            tx.wait()
            assert tx.is_completed is True

            # Subsequent waits should be no-ops
            tx.wait()
            assert tx.is_completed is True
            tx.wait()
            assert tx.is_completed is True

    def test_can_wait_reflects_handler_behavior(self) -> None:
        """Test that can_wait() correctly delegates to handler."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        # Tensor -> Block is always synchronous
        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            assert tx.can_wait() is True  # Synchronous transfer
            assert tx.is_completed is False  # But not completed until wait()

            tx.wait()
            assert tx.can_wait() is True  # Still can call wait()
            assert tx.is_completed is True  # Now completed


class TestCopyContextManagerExtraction:
    """Test that copy works with both raw blocks and context managers."""

    def test_copy_with_context_managers(self) -> None:
        """Test copy operations using context managers with Pipe."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_full_tile(42.0)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
        )
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
        )
        pipe = Pipe(1000, 1001)

        # Use context managers directly in copy calls
        with src_dfb.reserve() as src_ctx:
            # Pass context manager to copy
            tx = copy(source, src_ctx)
            tx.wait()

        # Copy through pipe using context managers
        with src_dfb.wait() as src_ctx:
            # WaitContext -> Pipe
            tx = copy(src_ctx, pipe)
            tx.wait()

        with dst_dfb.reserve() as dst_ctx:
            # Pipe -> ReserveContext
            tx = copy(pipe, dst_ctx)
            tx.wait()

        # Verify data was transferred
        result = make_zeros_tile()
        with dst_dfb.wait() as dst_ctx:
            tx = copy(dst_ctx, result)
            tx.wait()

        assert tensors_equal(result, source)

    def test_mixed_context_managers_and_tensors(self) -> None:
        """Test mixing context managers with raw tensors."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_full_tile(3.14)
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=2,
        )

        # Tensor -> Context manager
        with dfb.reserve() as ctx:
            tx = copy(source, ctx)
            tx.wait()

        # Context manager -> Tensor
        result = make_zeros_tile()
        with dfb.wait() as ctx:
            tx = copy(ctx, result)
            tx.wait()

        assert tensors_equal(result, source)


class TestCopyErrorConditions:
    """Test error conditions and edge cases in copy operations."""

    def test_copy_creates_transaction_immediately(self) -> None:
        """Test that copy() creates transaction immediately, not on wait()."""
        from python.sim.copy import copy, CopyTransaction

        set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=2,
        )

        with dfb.reserve() as block:
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
