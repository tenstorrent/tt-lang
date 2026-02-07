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
    make_full_tile,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
    tensors_equal,
)

from python.sim import ttnn
from python.sim.block import Block
from python.sim.cbapi import CBAPI
from python.sim.copyhandlers import (
    BlockToPipeHandler,
    BlockToTensorHandler,
    PipeToBlockHandler,
    TensorToBlockHandler,
    handler_registry,
)
from python.sim.typedefs import Pipe

if TYPE_CHECKING:
    pass


@pytest.fixture
def api():
    """Provide a fresh CBAPI instance for each test."""
    return CBAPI()


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


class TestCopyValidationErrors:
    """Test validation and error handling in copy handlers."""

    def test_non_2d_tensor_to_block_fails(self, api: "CBAPI") -> None:
        """Test that copying a non-2D tensor to Block raises ValueError."""
        import torch
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        # Create a 3D torch tensor
        torch_3d = torch.ones(32, 32, 32)
        tensor_3d = ttnn.Tensor(torch_3d)

        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with pytest.raises(ValueError, match="Tensor must be 2-dimensional"):
            with cb.reserve() as block:
                copy(tensor_3d, block)

    def test_tile_count_mismatch_tensor_to_block(self, api: "CBAPI") -> None:
        """Test that tile count mismatch raises ValueError (Tensor -> Block)."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        # 3 tiles in tensor but CB expects 2 tiles
        source = make_rand_tensor(96, 32)  # 3x1 tiles
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )

        with pytest.raises(
            ValueError, match="Tensor shape .* does not match.*Block shape"
        ):
            with cb.reserve() as block:
                copy(source, block)


class TestPipeErrorHandling:
    """Test error handling for pipe operations."""

    def test_pipe_receive_timeout_no_sender(self, api: "CBAPI") -> None:
        """Test that receiving from pipe with no sender is detected as deadlock."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy
        from python.sim.greenlet_scheduler import GreenletScheduler, set_scheduler

        # Create a minimal scheduler context for this test
        scheduler = GreenletScheduler()
        set_scheduler(scheduler)

        try:

            def test_thread() -> None:
                _set_current_thread_type(ThreadType.DM)

                # Use a unique pipe address to avoid interference
                pipe = Pipe(9999, 10000)
                cb = CircularBuffer(
                    element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
                )

                with cb.reserve() as block:
                    tx = copy(pipe, block)
                    tx.wait()

            scheduler.add_thread("test-dm", test_thread, ThreadType.DM)

            # With scheduler, waiting on pipe with no sender is detected as deadlock
            with pytest.raises(RuntimeError, match="Deadlock detected"):
                scheduler.run()
        finally:
            set_scheduler(None)

    def test_pipe_length_mismatch(self, api: "CBAPI") -> None:
        """Test that pipe receive fails when Block length doesn't match sent data."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        pipe = Pipe(5000, 5001)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(2, 1), buffer_factor=2, api=api
        )
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Send 2 tiles
        with src_cb.reserve() as src_block:
            tx_send = copy(make_rand_tensor(64, 32), src_block)
            tx_send.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Try to receive into 1-tile block
        with pytest.raises(
            ValueError,
            match="Destination Block length .* does not match pipe data length",
        ):
            with dst_cb.reserve() as dst_block:
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()


class TestPipeMulticast:
    """Test pipe multicast to multiple receivers."""

    def test_pipe_multiple_receivers(self, api: "CBAPI") -> None:
        """Test that pipe correctly handles multiple receivers."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)
        grid = (100, 100)  # Set grid context for pipe operations

        # Range covering 2 cores: (10,0) and (10,1)
        pipe = Pipe((10, 0), (10, slice(0, 2)))

        tile = make_full_tile(42.0)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        dst_cb1 = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        dst_cb2 = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Send data
        with src_cb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # First receiver
        result1 = make_zeros_tile()
        with dst_cb1.reserve() as dst_block:
            tx_recv1 = copy(pipe, dst_block)
            tx_recv1.wait()
        with dst_cb1.wait() as dst_block:
            tx = copy(dst_block, result1)
            tx.wait()
        assert tensors_equal(result1, tile)

        # Second receiver
        result2 = make_zeros_tile()
        with dst_cb2.reserve() as dst_block:
            tx_recv2 = copy(pipe, dst_block)
            tx_recv2.wait()
        with dst_cb2.wait() as dst_block:
            tx = copy(dst_block, result2)
            tx.wait()
        assert tensors_equal(result2, tile)


class TestTileCountUtility:
    """Test tile_count utility function."""

    def test_tile_count_basic(self) -> None:
        """Test basic tile counting."""
        from python.sim.copyhandlers import tile_count
        from python.sim.constants import TILE_SHAPE

        # 64x64 tensor with 32x32 tiles = 4 tiles (2x2 grid)
        assert tile_count((64, 64), TILE_SHAPE) == 4

        # 32x32 tensor with 32x32 tiles = 1 tile
        assert tile_count((32, 32), TILE_SHAPE) == 1

        # 96x64 tensor with 32x32 tiles = 6 tiles (3x2 grid)
        assert tile_count((96, 64), TILE_SHAPE) == 6

    def test_tile_count_dimension_mismatch(self) -> None:
        """Test that dimension mismatch raises ValueError."""
        from python.sim.copyhandlers import tile_count

        with pytest.raises(
            ValueError,
            match="tensor_shape and tile_shape must have same dimensions",
        ):
            tile_count((64, 64), (32,))  # 2D vs 1D

        with pytest.raises(
            ValueError,
            match="tensor_shape and tile_shape must have same dimensions",
        ):
            tile_count((64,), (32, 32))  # 1D vs 2D


class TestContextManagerHandlers:
    """Test context manager wrapper handler delegation."""

    def test_tensor_to_reserve_context(self, api: "CBAPI") -> None:
        """Test Tensor → ReserveContext handler delegation."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        source = make_full_tile(5.0)
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Read back and verify
        result = make_zeros_tile()
        with cb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        assert tensors_equal(result, source)

    def test_wait_context_to_tensor(self, api: "CBAPI") -> None:
        """Test WaitContext → Tensor handler delegation."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        source = make_full_tile(7.0)
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Write to CB
        with cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Read using context manager
        result = make_zeros_tile()
        with cb.wait() as block:
            tx = copy(block, result)
            tx.wait()

        assert tensors_equal(result, source)

    def test_pipe_to_reserve_context(self, api: "CBAPI") -> None:
        """Test Pipe → ReserveContext handler delegation."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        pipe = Pipe(7000, 7001)
        tile = make_full_tile(9.0)

        # Send data
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        with src_cb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive using ReserveContext
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        result = make_zeros_tile()
        with dst_cb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_cb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_wait_context_to_pipe(self, api: "CBAPI") -> None:
        """Test WaitContext → Pipe handler delegation."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        pipe = Pipe(8000, 8001)
        tile = make_full_tile(11.0)

        # Send using WaitContext
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        with src_cb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        result = make_zeros_tile()
        with dst_cb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_cb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_reserve_context_to_pipe(self, api: "CBAPI") -> None:
        """Test ReserveContext → Pipe handler delegation."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        pipe = Pipe(9000, 9001)
        tile = make_full_tile(13.0)

        # Send using ReserveContext
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        with src_cb.reserve() as src_block:
            tx1 = copy(tile, src_block)
            tx1.wait()
            # Note: Can't use reserve context as pipe source directly since
            # reserve() blocks are in WO state initially. Need to read from wait() instead.

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        result = make_zeros_tile()
        with dst_cb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_cb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)


class TestPipeCoreRangeTypes:
    """Test pipe multicast with different dst_core_range types."""

    def test_pipe_single_core_int(self, api: "CBAPI") -> None:
        """Test pipe with single 1D core (int)."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        # Single 1D core
        pipe = Pipe(0, 1)  # src_core=0, dst_core_range=1 (single int)

        tile = make_full_tile(15.0)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Send
        with src_cb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        result = make_zeros_tile()
        with dst_cb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_cb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_pipe_single_core_tuple(self, api: "CBAPI") -> None:
        """Test pipe with single multi-dimensional core (tuple)."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        # Single 2D core
        pipe = Pipe(
            (0, 0), (1, 1)
        )  # src_core=(0,0), dst_core_range=(1,1) (single tuple)

        tile = make_full_tile(17.0)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Send
        with src_cb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive
        result = make_zeros_tile()
        with dst_cb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            tx_recv.wait()

        with dst_cb.wait() as dst_block:
            tx = copy(dst_block, result)
            tx.wait()

        assert tensors_equal(result, tile)

    def test_pipe_core_range(self, api: "CBAPI") -> None:
        """Test pipe with core range (2x2 = 4 receivers)."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)
        grid = (100, 100)  # Set grid context for pipe operations

        # Core range: (20,20) to (21,21) = 2x2 = 4 cores
        pipe = Pipe((20, 20), (slice(20, 22), slice(20, 22)))

        tile = make_full_tile(19.0)
        src_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Send data
        with src_cb.reserve() as src_block:
            tx = copy(tile, src_block)
            tx.wait()

        with src_cb.wait() as src_block:
            tx_send = copy(src_block, pipe)
            tx_send.wait()

        # Receive from all 4 receivers
        for i in range(4):
            dst_cb = CircularBuffer(
                element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
            )
            result = make_zeros_tile()

            with dst_cb.reserve() as dst_block:
                tx_recv = copy(pipe, dst_block)
                tx_recv.wait()

            with dst_cb.wait() as dst_block:
                tx = copy(dst_block, result)
                tx.wait()

            assert tensors_equal(result, tile), f"Receiver {i} data mismatch"


class TestCanWaitBehavior:
    """Test can_wait() behavior for different handlers."""

    def test_tensor_to_block_can_wait_immediate(self, api: "CBAPI") -> None:
        """Test that Tensor → Block copy can_wait returns True immediately."""
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
            # can_wait should return True immediately for Tensor → Block
            assert tx.can_wait() is True
            tx.wait()

    def test_block_to_tensor_can_wait_immediate(self, api: "CBAPI") -> None:
        """Test that Block → Tensor copy can_wait returns True immediately."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        source = make_full_tile(21.0)
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Store data
        with cb.reserve() as block:
            tx = copy(source, block)
            tx.wait()

        # Copy to tensor
        result = make_zeros_tile()
        with cb.wait() as block:
            tx = copy(block, result)
            # can_wait should return True immediately for Block → Tensor
            assert tx.can_wait() is True
            tx.wait()

    def test_block_to_pipe_can_wait_immediate(self, api: "CBAPI") -> None:
        """Test that Block → Pipe copy can_wait returns True immediately."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        pipe = Pipe(11000, 11001)
        tile = make_full_tile(23.0)
        cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        # Store data
        with cb.reserve() as block:
            tx = copy(tile, block)
            tx.wait()

        # Send to pipe
        with cb.wait() as block:
            tx_send = copy(block, pipe)
            # can_wait should return True immediately for Block → Pipe
            assert tx_send.can_wait() is True
            tx_send.wait()

    def test_pipe_to_block_can_wait_blocks_until_data(self, api: "CBAPI") -> None:
        """Test that Pipe → Block copy can_wait blocks until data is available."""
        from python.sim.block import _set_current_thread_type, ThreadType
        from python.sim.cb import CircularBuffer
        from python.sim.copy import copy

        _set_current_thread_type(ThreadType.DM)

        pipe = Pipe(12000, 12001)
        dst_cb = CircularBuffer(
            element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
        )

        with dst_cb.reserve() as dst_block:
            tx_recv = copy(pipe, dst_block)
            # can_wait should return False before data is sent
            assert tx_recv.can_wait() is False

            # Now send data in a separate "thread" (simulated by just doing it)
            src_cb = CircularBuffer(
                element=make_ones_tile(), shape=(1, 1), buffer_factor=2, api=api
            )
            tile = make_full_tile(25.0)

            with src_cb.reserve() as src_block:
                tx_store = copy(tile, src_block)
                tx_store.wait()

            with src_cb.wait() as src_block:
                tx_send = copy(src_block, pipe)
                tx_send.wait()

            # Now can_wait should return True
            assert tx_recv.can_wait() is True
            tx_recv.wait()


if __name__ == "__main__":
    pytest.main([__file__])
