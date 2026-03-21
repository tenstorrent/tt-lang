# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for copy transfer handlers.

Tests the validation, error handling, and edge cases of copy handlers
using proper reserve()/wait() patterns conforming to the state machine.
"""

from typing import TYPE_CHECKING

import pytest
from test_utils import (
    make_element_for_buffer_shape,
    make_full_tile,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
    tensors_equal,
)

from python.sim import ttnn
from python.sim.blockstate import ThreadType
from python.sim.context import set_current_thread_type
from python.sim.dfb import Block, DataflowBuffer
from python.sim.copyhandlers import (
    BlockToPipeHandler,
    BlockToTensorHandler,
    PipeToBlockHandler,
    TensorToBlockHandler,
    HANDLER_REGISTRY,
)
from python.sim.pipe import Pipe

if TYPE_CHECKING:
    pass


@pytest.fixture(autouse=True)
def setup_scheduler_context(dm_thread_context):
    """Automatically set scheduler context for all copy handler tests.

    Copy operations typically happen in DM threads.
    """
    # Use the shared dm_thread_context fixture
    pass


class TestHandlerRegistry:
    """Test the handler registry mechanism."""

    def test_registry_populated(self):
        """Test that all handlers are registered."""
        assert (ttnn.Tensor, Block) in HANDLER_REGISTRY
        assert (Block, ttnn.Tensor) in HANDLER_REGISTRY
        assert (Block, Pipe) in HANDLER_REGISTRY
        assert (Pipe, Block) in HANDLER_REGISTRY

    def test_registry_handlers_correct_type(self):
        """Test that registered handlers are the correct instances."""
        assert isinstance(HANDLER_REGISTRY[(ttnn.Tensor, Block)], TensorToBlockHandler)
        assert isinstance(HANDLER_REGISTRY[(Block, ttnn.Tensor)], BlockToTensorHandler)
        assert isinstance(HANDLER_REGISTRY[(Block, Pipe)], BlockToPipeHandler)
        assert isinstance(HANDLER_REGISTRY[(Pipe, Block)], PipeToBlockHandler)


class TestCopyValidationErrors:
    """Test validation and error handling in copy handlers."""

    def test_nd_tensor_tile_count_mismatch_to_block_fails(self) -> None:
        """Test that an N-D tensor with mismatched total tile count raises ValueError."""
        import torch
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        # 3D tensor (2, 32, 32) has 2 total tiles; block (1, 1) has 1 total tile.
        torch_3d = torch.ones(2, 32, 32)
        tensor_3d = ttnn.Tensor(torch_3d)

        dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        with pytest.raises(ValueError, match="does not match"):
            with dfb.reserve() as block:
                copy(tensor_3d, block)

    def test_tile_count_mismatch_tensor_to_block(self) -> None:
        """Test that tile count mismatch raises ValueError (Tensor -> Block)."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        # 3 tiles in tensor but DFB expects 2 tiles
        source = make_rand_tensor(96, 32)  # 3x1 tiles
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            buffer_factor=2,
        )

        with pytest.raises(
            ValueError, match="Tensor shape .* does not match.*Block shape"
        ):
            with dfb.reserve() as block:
                copy(source, block)


class TestPipeErrorHandling:
    """Test error handling for pipe operations."""

    def test_pipe_receive_timeout_no_sender(self) -> None:
        """Test that receiving from pipe with no sender is detected as deadlock."""
        from python.sim.copy import copy
        from python.sim.greenlet_scheduler import GreenletScheduler, set_scheduler

        # Create a minimal scheduler context for this test
        scheduler = GreenletScheduler()
        set_scheduler(scheduler)

        try:

            def test_thread() -> None:
                set_current_thread_type(ThreadType.DM)

                # Use a unique pipe address to avoid interference
                pipe = Pipe(9999, 10000)
                dfb = DataflowBuffer(
                    likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
                )

                with dfb.reserve() as block:
                    tx = copy(pipe, block)
                    tx.wait()

            scheduler.add_thread("test-dm", test_thread, ThreadType.DM)

            # With scheduler, waiting on pipe with no sender is detected as deadlock
            with pytest.raises(RuntimeError, match="Deadlock detected"):
                scheduler.run()
        finally:
            set_scheduler(None)

    def test_pipe_length_mismatch(self) -> None:
        """Test that pipe receive fails when Block length doesn't match sent data."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        pipe = Pipe(5000, 5001)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            buffer_factor=2,
        )
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            buffer_factor=2,
        )

        # Send 2 tiles
        with src_dfb.reserve() as src_block:
            tx_send = copy(make_rand_tensor(64, 32), src_block)
            tx_send.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Try to receive into 1-tile block
        with pytest.raises(
            ValueError,
            match="Destination Block shape .* does not match pipe data shape",
        ):
            with dst_dfb.reserve() as dst_block:
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()


class TestPipeMulticast:
    """Test pipe multicast to multiple receivers."""

    def test_pipe_multiple_receivers(self) -> None:
        """Test that pipe correctly handles multiple receivers."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)
        grid = (100, 100)  # Set grid context for pipe operations

        # Range covering 2 cores: (10,0) and (10,1)
        pipe = Pipe((10, 0), (10, slice(0, 2)))

        tile = make_full_tile(42.0)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        dst_dfb1 = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        dst_dfb2 = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        # Send data
        with src_dfb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # First receiver
        result1 = make_zeros_tile()
        with dst_dfb1.reserve() as dst_block:
            tx_recv1 = copy(pipe, dst_block)
            tx_recv1.wait()
        with dst_dfb1.wait() as dst_block:
            tx = copy(dst_block, result1)
            tx.wait()
        assert tensors_equal(result1, tile)

        # Second receiver
        result2 = make_zeros_tile()
        with dst_dfb2.reserve() as dst_block:
            tx_recv2 = copy(pipe, dst_block)
            tx_recv2.wait()
        with dst_dfb2.wait() as dst_block:
            tx = copy(dst_block, result2)
            tx.wait()
        assert tensors_equal(result2, tile)


class TestContextManagerHandlers:
    """Test context manager wrapper handler delegation."""

    def test_tensor_to_reserve_context(self) -> None:
        """Test Tensor → ReserveContext handler delegation."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_full_tile(5.0)
        dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Read back and verify
        result = make_zeros_tile()
        with dfb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        assert tensors_equal(result, source)

    def test_wait_context_to_tensor(self) -> None:
        """Test WaitContext → Tensor handler delegation."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_full_tile(7.0)
        dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        # Write to DFB
        with dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Read using context manager
        result = make_zeros_tile()
        with dfb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        assert tensors_equal(result, source)

    def test_pipe_to_reserve_context(self) -> None:
        """Test Pipe → ReserveContext handler delegation."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        pipe = Pipe(7000, 7001)
        tile = make_full_tile(9.0)

        # Send data
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        with src_dfb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive using ReserveContext
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        result = make_zeros_tile()
        with dst_dfb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_dfb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_wait_context_to_pipe(self) -> None:
        """Test WaitContext → Pipe handler delegation."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        pipe = Pipe(8000, 8001)
        tile = make_full_tile(11.0)

        # Send using WaitContext
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        with src_dfb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        result = make_zeros_tile()
        with dst_dfb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_dfb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_reserve_context_to_pipe(self) -> None:
        """Test ReserveContext → Pipe handler delegation."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        pipe = Pipe(9000, 9001)
        tile = make_full_tile(13.0)

        # Send using ReserveContext
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        with src_dfb.reserve() as src_block:
            tx1 = copy(tile, src_block)
            tx1.wait()
            # Note: Can't use reserve context as pipe source directly since
            # reserve() blocks are in WO state initially. Need to read from wait() instead.

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        result = make_zeros_tile()
        with dst_dfb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_dfb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)


class TestPipeCoreRangeTypes:
    """Test pipe multicast with different dst_core_range types."""

    def test_pipe_single_node_int(self) -> None:
        """Test pipe with single 1D core (int)."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        # Single 1D core
        pipe = Pipe(0, 1)  # src_core=0, dst_core_range=1 (single int)

        tile = make_full_tile(15.0)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        # Send
        with src_dfb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        result = make_zeros_tile()
        with dst_dfb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_dfb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_pipe_single_node_tuple(self) -> None:
        """Test pipe with single multi-dimensional core (tuple)."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        # Single 2D core
        pipe = Pipe(
            (0, 0), (1, 1)
        )  # src_core=(0,0), dst_core_range=(1,1) (single tuple)

        tile = make_full_tile(17.0)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        # Send
        with src_dfb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        result = make_zeros_tile()
        with dst_dfb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_dfb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_pipe_core_range(self) -> None:
        """Test pipe with core range (2x2 = 4 receivers)."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)
        grid = (100, 100)  # Set grid context for pipe operations

        # Core range: (20,20) to (21,21) = 2x2 = 4 cores
        pipe = Pipe((20, 20), (slice(20, 22), slice(20, 22)))

        tile = make_full_tile(19.0)
        src_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        # Send data
        with src_dfb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_dfb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive from all 4 receivers
        for i in range(4):
            dst_dfb = DataflowBuffer(
                likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
            )
            result = make_zeros_tile()

            with dst_dfb.reserve() as dst_block:
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()

            with dst_dfb.wait() as dst_block:
                tx = copy(dst_block, result)
                tx.wait()

            assert tensors_equal(result, tile), f"Receiver {i} data mismatch"


class TestCanWaitBehavior:
    """Test can_wait() behavior for different handlers."""

    def test_tensor_to_block_can_wait_immediate(self) -> None:
        """Test that Tensor → Block copy can_wait returns True immediately."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_ones_tile()
        dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        with dfb.reserve() as block:
            tx = copy(source, block)
            # can_wait should return True immediately for Tensor → Block
            assert tx.can_wait() is True
            tx.wait()

    def test_block_to_tensor_can_wait_immediate(self) -> None:
        """Test that Block → Tensor copy can_wait returns True immediately."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        source = make_full_tile(21.0)
        dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        # Store data
        with dfb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Copy to tensor
        result = make_zeros_tile()
        with dfb.wait() as block:
            tx = copy(block, result)
            # can_wait should return True immediately for Block → Tensor
            assert tx.can_wait() is True
            tx.wait()

    def test_block_to_pipe_can_wait_immediate(self) -> None:
        """Test that Block → Pipe copy can_wait returns True immediately."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        pipe = Pipe(11000, 11001)
        tile = make_full_tile(23.0)
        dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        # Store data
        with dfb.reserve() as block:
            tx = copy(tile, block)
            tx.wait()

        # Send to pipe
        with dfb.wait() as block:
            tx_send = copy(block, pipe)
            # can_wait should return True immediately for Block → Pipe
            assert tx_send.can_wait() is True
            tx_send.wait()

    def test_pipe_to_block_can_wait_blocks_until_data(self) -> None:
        """Test that Pipe → Block copy can_wait blocks until data is available."""
        from python.sim.copy import copy

        set_current_thread_type(ThreadType.DM)

        pipe = Pipe(12000, 12001)
        dst_dfb = DataflowBuffer(
            likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
        )

        with dst_dfb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            # can_wait should return False before data is sent
            assert tx_recv.can_wait() is False

            # Now send data in a separate "thread" (simulated by just doing it)
            src_dfb = DataflowBuffer(
                likeness_tensor=make_ones_tile(), shape=(1, 1), buffer_factor=2
            )
            tile = make_full_tile(25.0)

            with src_dfb.reserve() as src_block:
                tx_store = copy(tile, src_block)
                tx_store.wait()

            with src_dfb.wait() as src_block:
                tx_send = copy(src_block, pipe)
                tx_send.wait()

            # Now can_wait should return True
            assert tx_recv.can_wait() is True
            tx_recv.wait()


if __name__ == "__main__":
    pytest.main([__file__])
