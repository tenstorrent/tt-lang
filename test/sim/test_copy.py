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
from test_utils import (
    make_element_for_buffer_shape,
    make_full_tile,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
    tensors_equal,
)

from sim.blockstate import BlockAcquisition
from sim.context import set_current_kernel_type
from sim.dfb import Block, DataflowBuffer
from sim.ttnnsim import ROW_MAJOR_LAYOUT, Tensor
from sim.copy import CopyTransaction, GroupTransfer, copy
from sim.pipe import Pipe
from sim.kernel import KernelKind


@pytest.fixture(autouse=True)
def setup_scheduler_context(dm_kernel_context):
    """Automatically set scheduler context for all copy tests.

    Copy operations typically happen in DM kernels.
    """
    # Use the shared dm_kernel_context fixture
    pass


class TestCopyTransaction:
    """Test CopyTransaction class functionality."""

    @pytest.mark.parametrize(
        "byte_count",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
            pytest.param(True, id="boolean"),
            pytest.param(1.5, id="non-integer"),
        ],
    )
    def test_byte_count_must_be_a_positive_integer(self, byte_count: object) -> None:
        """Reject values that do not denote a positive byte count."""

        with pytest.raises(ValueError, match="must be a positive int"):
            CopyTransaction(
                make_ones_tile(),
                make_zeros_tile(),
                byte_count=byte_count,
            )

    def test_byte_count_must_fit_noc_transfer_size(self) -> None:
        """Reject counts that the TTKernel NoC size operand cannot represent."""

        with pytest.raises(ValueError, match="unsigned 32-bit NoC transfer size"):
            CopyTransaction(
                make_ones_tile(),
                make_zeros_tile(),
                byte_count=1 << 32,
            )

    @pytest.mark.parametrize(
        ("source_shape", "destination_shape", "endpoint"),
        [
            pytest.param((1, 1), (2, 1), "source", id="source"),
            pytest.param((2, 1), (1, 1), "destination", id="destination"),
        ],
    )
    def test_byte_count_must_fit_each_dfb_block(
        self,
        source_shape: tuple[int, int],
        destination_shape: tuple[int, int],
        endpoint: str,
    ) -> None:
        """Reject a count that exceeds either acquired DFB block."""

        source_tensor = make_element_for_buffer_shape(source_shape)
        destination_tensor = make_element_for_buffer_shape(destination_shape)
        source_dfb = DataflowBuffer(
            likeness_tensor=source_tensor, shape=source_shape, block_count=1
        )
        destination_dfb = DataflowBuffer(
            likeness_tensor=destination_tensor,
            shape=destination_shape,
            block_count=1,
        )
        source_block = Block(
            tensor=source_tensor,
            shape=source_shape,
            acquisition=BlockAcquisition.WAIT,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=source_dfb,
        )
        destination_block = Block(
            tensor=destination_tensor,
            shape=destination_shape,
            acquisition=BlockAcquisition.RESERVE,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=destination_dfb,
        )

        with pytest.raises(ValueError, match=f"exceeds {endpoint} capacity"):
            CopyTransaction(source_block, destination_block, byte_count=2049)

    def test_byte_count_requires_tiled_dfb_blocks(self) -> None:
        """Reject byte-counted transfers from row-major DFB storage."""

        source_tensor = Tensor(torch.zeros(8, dtype=torch.float32), ROW_MAJOR_LAYOUT)
        source_dfb = DataflowBuffer(
            likeness_tensor=source_tensor, shape=(8,), block_count=1
        )
        source_block = Block(
            source_tensor,
            shape=(8,),
            acquisition=BlockAcquisition.WAIT,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=source_dfb,
        )

        with pytest.raises(ValueError, match="must use TILE layout"):
            CopyTransaction(source_block, Pipe(7000, 7001), byte_count=4)

    def test_copy_transaction_unsupported_types(self) -> None:
        """Test that unsupported type combinations raise ValueError."""
        tensor1 = make_rand_tensor(32, 32)
        tensor2 = make_zeros_tile()

        # tensor → tensor not supported
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Tensor, Tensor\\)"
        ):
            CopyTransaction(tensor1, tensor2)


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
            kernel_type=KernelKind.DATA_MOVEMENT,
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
            kernel_type=KernelKind.DATA_MOVEMENT,
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
            kernel_type=KernelKind.DATA_MOVEMENT,
        )

        # Create destination tensor (non-Block, so no state changes)
        dest_tensor = make_rand_tensor(64, 32)

        # Start copy
        tx = copy(source_block, dest_tensor)

        # Attempt to write to source block should fail (source expects TX_WAIT)
        # But more fundamentally, wait() blocks don't support store() - they expect POP
        with pytest.raises(
            RuntimeError,
            match=r"(?s)Cannot write to this buffer block.*ROR",
        ):
            source_block.store(Block.from_tensor(make_rand_tensor(64, 32)))

        # After wait(), the block still doesn't support store() because it's a wait() block
        tx.wait()
        # wait() blocks cannot use store() per state machine - they expect STORE_SRC
        with pytest.raises(
            RuntimeError,
            match=r"(?s)Cannot perform store\(\): not a valid next dataflow step.*expected one of",
        ):
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
            kernel_type=KernelKind.DATA_MOVEMENT,
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
            kernel_type=KernelKind.DATA_MOVEMENT,
        )

        # Start copy
        tx = copy(source_tensor, dest_block)

        # Attempt to write to destination should fail (dest is in NAW state)
        with pytest.raises(
            RuntimeError,
            match=r"(?s)Cannot write to this buffer block.*NAW.*copy lock error",
        ):
            dest_block.store(Block.from_tensor(make_rand_tensor(64, 32)))

        # After wait(), block expects PUSH (not store) per state machine
        tx.wait()
        # Cannot store on DM block - only Compute blocks support store
        with pytest.raises(
            RuntimeError,
            match=r"(?s)Cannot perform store\(\): not a valid next dataflow step.*expected one of",
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
            kernel_type=KernelKind.DATA_MOVEMENT,
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
            match=r"(?s)Cannot perform copy \(as destination\): not a valid next dataflow step.*\[COPY_SRC, TX_WAIT\].*attempted COPY_DST",
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
        """Test Tensor -> Block copy using reserve() in DM kernel."""

        # Set DM kernel context for copy operations
        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        """Test Block -> Tensor copy using wait() in DM kernel."""

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

    def test_byte_counted_block_copy_preserves_destination_tail(self) -> None:
        """Copy the declared number of bytes from the start of each DFB block."""
        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

        source = make_rand_tensor(64, 32)
        source_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=1,
        )
        destination_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=1,
        )

        with source_dfb.reserve() as source_block:
            copy(source, source_block).wait()

        with source_dfb.wait() as source_block:
            with destination_dfb.reserve() as destination_block:
                destination_block.raw_tensor.to_torch().fill_(-7.0)
                bytes_per_tile = source_block.raw_tensor.size_in_bytes(32 * 32)
                copy(
                    source_block,
                    destination_block,
                    byte_count=bytes_per_tile,
                ).wait()

                result = destination_block.raw_tensor.to_torch().reshape(-1)
                expected = source.to_torch().reshape(-1)
                assert torch.equal(result[: 32 * 32], expected[: 32 * 32])
                assert torch.all(result[32 * 32 :] == -7.0)

    def test_byte_counted_block_copy_preserves_partial_element_bytes(
        self,
    ) -> None:
        """Model the operation's byte granularity, including partial elements."""
        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

        source = make_rand_tensor(32, 32)
        source_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )
        destination_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )

        with source_dfb.reserve() as source_block:
            copy(source, source_block).wait()

        with source_dfb.wait() as source_block:
            with destination_dfb.reserve() as destination_block:
                destination_block.raw_tensor.to_torch().fill_(-7.0)
                source_values = source_block.raw_tensor.to_torch().reshape(-1)
                expected = destination_block.raw_tensor.to_torch().reshape(-1).clone()
                expected[0] = source_values[0]
                source_partial = source_values[1:2].to(
                    dtype=source_block.raw_tensor.dtype
                )
                expected_partial = expected[1:2].to(
                    dtype=destination_block.raw_tensor.dtype
                )
                source_partial_bytes = source_partial.view(dtype=torch.uint8).reshape(
                    -1
                )
                expected_partial_bytes = expected_partial.view(
                    dtype=torch.uint8
                ).reshape(-1)
                expected_partial_bytes[0] = source_partial_bytes[0]
                expected[1] = expected_partial.to(
                    dtype=destination_block.raw_tensor.underlying_dtype
                ).item()

                copy(source_block, destination_block, byte_count=3).wait()

                destination_after = destination_block.raw_tensor.to_torch().reshape(-1)
                assert torch.equal(destination_after, expected)

    def test_block_copy_requires_byte_count(self) -> None:
        """Reject block-to-block copies without an explicit byte count."""
        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

        source = make_rand_tensor(32, 32)
        source_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )
        destination_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )

        with source_dfb.reserve() as source_block:
            copy(source, source_block).wait()

        with source_dfb.wait() as source_block:
            with destination_dfb.reserve() as destination_block:
                with pytest.raises(
                    ValueError, match="Block-to-block copy requires byte_count"
                ):
                    copy(source_block, destination_block)
                copy(source_block, destination_block, byte_count=64).wait()

    @pytest.mark.parametrize(
        ("source_acquisition", "destination_acquisition", "expected_error"),
        [
            pytest.param(
                BlockAcquisition.RESERVE,
                BlockAcquisition.RESERVE,
                "source must come from DFB wait",
                id="source-not-waited",
            ),
            pytest.param(
                BlockAcquisition.WAIT,
                BlockAcquisition.WAIT,
                "destination must come from DFB reserve",
                id="destination-not-reserved",
            ),
        ],
    )
    def test_byte_counted_block_copy_requires_acquired_directions(
        self,
        source_acquisition,
        destination_acquisition,
        expected_error,
    ) -> None:
        """Reject block copies that reverse the producer/consumer roles."""
        source_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=1
        )
        destination_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=1
        )
        source_block = Block(
            make_ones_tile(),
            shape=(1, 1),
            acquisition=source_acquisition,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=source_dfb,
        )
        destination_block = Block(
            make_zeros_tile(),
            shape=(1, 1),
            acquisition=destination_acquisition,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=destination_dfb,
        )

        with pytest.raises(ValueError, match=expected_error):
            CopyTransaction(source_block, destination_block, byte_count=64)

    def test_byte_counted_block_copy_requires_distinct_dfbs(self) -> None:
        """Reject local byte copies within one DFB lifecycle."""
        dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2
        )
        source_block = Block(
            make_ones_tile(),
            shape=(1, 1),
            acquisition=BlockAcquisition.WAIT,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=dfb,
        )
        destination_block = Block(
            make_zeros_tile(),
            shape=(1, 1),
            acquisition=BlockAcquisition.RESERVE,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=dfb,
        )

        with pytest.raises(
            ValueError, match="requires distinct source and destination"
        ):
            CopyTransaction(source_block, destination_block, byte_count=64)

    def test_byte_counted_pipe_copy_requires_matching_counts(self) -> None:
        """Reject a receiver count that differs from the queued send count."""
        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

        source_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=1
        )
        destination_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=1
        )
        pipe = Pipe(214, 215)
        with source_dfb.reserve() as source_block:
            copy(make_ones_tile(), source_block).wait()
        with source_dfb.wait() as source_block:
            copy(source_block, pipe, byte_count=64).wait()

        destination_block = Block(
            make_zeros_tile(),
            shape=(1, 1),
            acquisition=BlockAcquisition.RESERVE,
            kernel_type=KernelKind.DATA_MOVEMENT,
            dfb=destination_dfb,
        )
        transaction = CopyTransaction(pipe, destination_block, byte_count=32)
        with pytest.raises(ValueError, match="sender and receiver must use the same"):
            transaction.wait()

    def test_byte_counted_pipe_copy_preserves_destination_tail(self) -> None:
        """Use the same byte count on pipe send and receive."""
        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

        source = make_rand_tensor(64, 32)
        source_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=1,
        )
        destination_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=1,
        )
        pipe = Pipe(212, 213)

        with source_dfb.reserve() as source_block:
            copy(source, source_block).wait()

        with source_dfb.wait() as source_block:
            with destination_dfb.reserve() as destination_block:
                destination_block.raw_tensor.to_torch().fill_(-11.0)
                bytes_per_tile = source_block.raw_tensor.size_in_bytes(32 * 32)
                copy(source_block, pipe, byte_count=bytes_per_tile).wait()
                copy(pipe, destination_block, byte_count=bytes_per_tile).wait()

                result = destination_block.raw_tensor.to_torch().reshape(-1)
                expected = source.to_torch().reshape(-1)
                assert torch.equal(result[: 32 * 32], expected[: 32 * 32])
                assert torch.all(result[32 * 32 :] == -11.0)

    def test_copy_with_pipe_single_tile(self) -> None:
        """Test Block -> Pipe -> Block copy with single tile."""

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)
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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        from sim.copy import copy

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        from sim.copy import copy

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        from sim.copy import copy

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        from sim.copy import copy

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        from sim.copy import copy

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        from sim.copy import copy, CopyTransaction

        set_current_kernel_type(KernelKind.DATA_MOVEMENT)

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
        from sim.copy import copy

        tensor1 = make_ones_tile()
        tensor2 = make_zeros_tile()

        # Tensor -> Tensor is not supported
        with pytest.raises(
            ValueError, match="No copy handler registered for \\(Tensor, Tensor\\)"
        ):
            copy(tensor1, tensor2)


class TestGroupTransfer:
    """Tests for GroupTransfer: grouping and waiting on multiple copy handles."""

    def test_wait_all_completes_all_transfers(self) -> None:
        """wait_all() executes all transfers in the group."""
        source = make_rand_tensor(32, 32)
        dfb1 = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )
        dfb2 = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )

        with dfb1.reserve() as blk1, dfb2.reserve() as blk2:
            gxf = GroupTransfer()
            gxf.add(copy(source, blk1))
            gxf.add(copy(source, blk2))
            gxf.wait_all()

            assert tensors_equal(blk1.to_tensor(), source)
            assert tensors_equal(blk2.to_tensor(), source)

    def test_add_after_wait_all_raises(self) -> None:
        """add() after wait_all() raises RuntimeError."""
        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )

        with dfb.reserve() as blk:
            gxf = GroupTransfer()
            gxf.add(copy(source, blk))
            gxf.wait_all()

        # add() checks _waited before touching its argument, so a sentinel suffices.
        with pytest.raises(RuntimeError, match="after wait_all"):
            gxf.add(None)  # type: ignore[arg-type]

    def test_wait_all_twice_raises(self) -> None:
        """Calling wait_all() twice raises RuntimeError."""
        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            block_count=1,
        )

        with dfb.reserve() as blk:
            gxf = GroupTransfer()
            gxf.add(copy(source, blk))
            gxf.wait_all()

        with pytest.raises(RuntimeError, match="more than once"):
            gxf.wait_all()

    def test_empty_group_wait_all(self) -> None:
        """wait_all() on an empty group completes without error."""
        gxf = GroupTransfer()
        gxf.wait_all()  # should not raise


if __name__ == "__main__":
    pytest.main([__file__])
