# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for DataflowBuffer.

Covers the high-level DataflowBuffer interface (tensor-aware operations,
context manager syntax, state machine enforcement) and the low-level ring-buffer
primitives (reserve/wait/push/pop, error contracts, per-core limits).
"""

import pytest
import subprocess
import tempfile
from pathlib import Path
import torch
from test_utils import (
    make_element_for_buffer_shape,
    make_full_tensor,
    make_ones_tensor,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tensor,
    make_zeros_tile,
    tensors_equal,
    tensors_exact_equal,
)

from python.sim import TILE_SHAPE, copy, ttnn
from python.sim.ttnnsim import Tensor
from python.sim.dfb import (
    Block,
    DataflowBuffer,
)
from python.sim.math import broadcast
from python.sim.blockstate import (
    ThreadType,
    BlockAcquisition,
)
from python.sim.context import (
    set_current_thread_type,
    clear_current_thread_type,
)


@pytest.fixture(autouse=True)
def setup_thread_context(compute_thread_context):
    """Automatically set thread context and scheduler for all DFB tests.

    Note: These tests primarily exercise COMPUTE thread patterns (using store()).
    DM thread patterns (using copy operations) are tested separately in copy/pipe tests.
    The state machine enforces different expected operations for DM vs COMPUTE threads,
    so parametrizing these tests would require substantial test logic changes.
    """
    # Use the shared compute_thread_context fixture
    pass


@pytest.fixture
def configured_dfb() -> DataflowBuffer:
    """Create a DataflowBuffer with capacity 4 (shape=(1,1), buffer_factor=4)."""
    element = make_ones_tile()
    return DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=4)


@pytest.fixture
def configured_dfb8() -> DataflowBuffer:
    """Create a DataflowBuffer with capacity 8 (shape=(1,1), buffer_factor=8)."""
    element = make_ones_tile()
    return DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=8)


def test_dataflow_buffer_basic() -> None:
    """Test basic DataflowBuffer operations."""
    element = make_ones_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

    assert dfb.shape == (1, 1)
    assert dfb.capacity_tiles == 2
    assert dfb.buffer_factor == 2

    write_view = dfb.reserve()
    assert len(write_view) == 1

    test_data = make_ones_tile()
    write_view.store(Block.from_tensor(test_data))
    write_view.push()

    read_view = dfb.wait()
    assert len(read_view) == 1

    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    out_block = out_dfb.reserve()
    out_block.store(read_view)
    out_block.push()

    read_data = read_view.to_list()
    assert read_data[0] is not None
    assert tensors_equal(read_data[0], test_data)

    read_view.pop()

    print("Basic DataflowBuffer test passed!")


def test_dataflow_buffer_multi_tile() -> None:
    """Test DataflowBuffer with multiple tiles per operation."""
    element = make_element_for_buffer_shape((2, 1))
    dfb = DataflowBuffer(likeness_tensor=element, shape=(2, 1), buffer_factor=3)

    assert dfb.shape == (2, 1)
    assert dfb.capacity_tiles == 6

    # Test reserve/push
    write_view = dfb.reserve()
    assert len(write_view) == 2  # Should have space for 2 tiles

    # Fill with test data: assemble two (32, 32) tiles into one (64, 32) tensor
    import torch as _torch

    tile0 = ttnn.rand(TILE_SHAPE)
    tile0.to_torch().fill_(1.0)
    tile1 = ttnn.rand(TILE_SHAPE)
    tile1.to_torch().fill_(2.0)
    assembled = ttnn.Tensor(_torch.cat([tile0.to_torch(), tile1.to_torch()], dim=0))
    write_view.store(Block.from_tensor(assembled))

    write_view.push()

    # Test wait/pop
    read_view = dfb.wait()
    assert len(read_view) == 2  # Should have 2 tiles available

    # Use waited block as source (STORE_SRC) before pop
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(2, 1), buffer_factor=2)
    out_block = out_dfb.reserve()
    out_block.store(read_view)
    out_block.push()

    read_data = read_view.to_list()
    for i in range(2):
        assert read_data[i] is not None
        expected_value = float(i + 1)
        actual_value = read_data[i].to_torch()[0, 0].item()
        assert (
            abs(actual_value - expected_value) < 1e-5
        ), f"Tile {i}: expected {expected_value}, got {actual_value}"

    read_view.pop()

    print("Multi-tile DataflowBuffer test passed!")


def test_copy_operations_with_dm_context() -> None:
    """Test copy operations between tensor and DataflowBuffer with proper DM thread context."""
    set_current_thread_type(ThreadType.DM)

    try:
        tensor_a = make_rand_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

        element = make_ones_tile()
        dfb_a = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

        # Test copy from tensor to dataflow buffer (DM thread can do this)
        dfb_view = dfb_a.reserve()
        tensor_slice = tensor_a[0:1, 0:1]  # Single tile

        # Copy operation
        tx = copy(tensor_slice, dfb_view)
        tx.wait()
        dfb_view.push()

        # Test copy from dataflow buffer back to tensor
        dfb_read_view = dfb_a.wait()
        output_tensor = make_zeros_tile()  # Single tile output

        # Copy operation
        tx2 = copy(dfb_read_view, output_tensor)
        tx2.wait()
        dfb_read_view.pop()

        # Verify the data was transferred
        assert output_tensor.shape == TILE_SHAPE
        # The output tensor should now contain the data from the dataflow buffer
        # Verify at least some data was copied (non-zero)
        import torch

        assert output_tensor.to_torch().sum() != 0

    finally:
        # Clean up thread context
        clear_current_thread_type()

    print("Copy operations with DM context test passed!")


def test_error_handling() -> None:
    """Test error conditions."""
    element = make_ones_tile()
    with pytest.raises(ValueError):
        DataflowBuffer(likeness_tensor=element, shape=(0, 1))  # Invalid shape element

    with pytest.raises(ValueError):
        DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=0)

    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

    # # Can't push without reserve - DFBAPI will catch this
    # with pytest.raises(DFBContractError):
    #     dfb.push()

    # # Can't pop without wait - DFBAPI will catch this
    # with pytest.raises(DFBContractError):
    #     dfb.pop()

    # Test unsupported copy operations with wrong types
    with pytest.raises(ValueError, match="No copy handler registered"):
        copy("invalid_source", "invalid_dest")  # type: ignore

    print("Error handling test passed!")


def test_copy_in_dm_thread_context() -> None:
    """Test copy operations with proper DM thread context.

    This test demonstrates the full workflow:
    - DM thread: copy data into DFBs (reserve + copy + push)
    - Switch to COMPUTE thread for consumption (wait + read + pop)
    """

    try:
        rows, cols = 128, 128
        granularity = 4

        a_in = make_rand_tensor(rows, cols)
        c_in = make_rand_tensor(TILE_SHAPE[0], cols)

        a_in_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((granularity, 1)),
            shape=(granularity, 1),
            buffer_factor=2,
        )
        c_in_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            buffer_factor=2,
        )

        # Verify the dataflow buffers were created correctly
        assert a_in_dfb.shape == (granularity, 1)
        assert a_in_dfb.capacity_tiles == granularity * 2
        assert c_in_dfb.shape == (1, 1)
        assert c_in_dfb.capacity_tiles == 2

        # DM thread: Producer side - copy data into DFBs
        set_current_thread_type(ThreadType.DM)

        # Copy c_in data
        c_block = c_in_dfb.reserve()
        c_slice = c_in[0:1, 0:1]
        tx = copy(c_slice, c_block)
        tx.wait()
        c_block.push()

        # Copy a_in data
        a_block = a_in_dfb.reserve()
        a_slice = a_in[0:granularity, 0:1]
        tx = copy(a_slice, a_block)
        tx.wait()
        a_block.push()

        # Switch to COMPUTE thread: Consumer side - read data back
        set_current_thread_type(ThreadType.COMPUTE)

        c_data = c_in_dfb.wait()
        a_data = a_in_dfb.wait()

        # Verify we got the expected views
        assert len(c_data) == 1
        assert len(a_data) == granularity

        # Verify data was copied correctly
        c_list = c_data.to_list()
        a_list = a_data.to_list()
        assert c_list[0] is not None
        assert a_list[0] is not None

        out_dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((1, 1)),
            shape=(1, 1),
            buffer_factor=2,
        )
        out_block = out_dfb.reserve()
        out_block.store(c_data)
        out_block.push()
        c_data.pop()

        out_dfb2 = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((granularity, 1)),
            shape=(granularity, 1),
            buffer_factor=2,
        )
        out_block2 = out_dfb2.reserve()
        out_block2.store(a_data)
        out_block2.push()
        a_data.pop()

    finally:
        # Clean up thread context
        clear_current_thread_type()

    print("Copy in DM thread context test passed!")


def test_single_pending_reserve_constraint() -> None:
    """Test that only one reserve() is allowed before push()."""
    from python.sim.copy import copy

    set_current_thread_type(ThreadType.DM)

    try:
        element = make_ones_tile()
        dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

        # Create a source tensor for copy operations
        src_tensor = make_ones_tile()

        # First reserve() should succeed
        block1 = dfb.reserve()
        assert block1 is not None

        # Second reserve() before push() should fail
        with pytest.raises(
            RuntimeError, match="Cannot call reserve\\(\\) again before push\\(\\)"
        ):
            dfb.reserve()

        # Complete the copy operation and push to get to PUSH state
        tx = copy(src_tensor, block1)
        tx.wait()

        # After push(), should be able to reserve() again
        block1.push()
        block2 = dfb.reserve()
        assert block2 is not None

        # Complete second block's operations
        tx = copy(src_tensor, block2)
        tx.wait()
        block2.push()
    finally:

        clear_current_thread_type()


def test_single_pending_wait_constraint() -> None:
    """Test that only one wait() is allowed before pop()."""
    from python.sim.copy import copy

    set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_ones_tile()
        dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

        set_current_thread_type(ThreadType.DM)
        block = dfb.reserve()
        test_data = make_rand_tensor(TILE_SHAPE[0], TILE_SHAPE[1])
        test_slice = test_data[0:1, 0:1]
        tx = copy(test_slice, block)
        tx.wait()
        block.push()

        # Switch to COMPUTE thread for consumption
        set_current_thread_type(ThreadType.COMPUTE)

        # First wait() should succeed
        data1 = dfb.wait()
        assert data1 is not None

        # Second wait() before pop() should fail
        with pytest.raises(
            RuntimeError, match="Cannot call wait\\(\\) again before pop\\(\\)"
        ):
            dfb.wait()

        out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
        out_block = out_dfb.reserve()
        out_block.store(data1)
        out_block.push()
        data1.pop()

        # Add more data (using DM thread)
        set_current_thread_type(ThreadType.DM)
        block = dfb.reserve()
        tx = copy(test_slice, block)
        tx.wait()
        block.push()

        set_current_thread_type(ThreadType.COMPUTE)
        data2 = dfb.wait()
        assert data2 is not None
        # Use second waited block as STORE_SRC before pop
        out_block2 = out_dfb.reserve()
        out_block2.store(data2)
        out_block2.push()
        data2.pop()
    finally:
        clear_current_thread_type()


def test_reserve_store_push_pop_workflow() -> None:
    """Test the complete reserve->store->push->wait->pop workflow."""
    import torch
    from python.sim import ttnn, TILE_SHAPE

    element = make_element_for_buffer_shape((2, 1))
    dfb = DataflowBuffer(likeness_tensor=element, shape=(2, 1), buffer_factor=2)

    with dfb.reserve() as write_block:
        write_block.store(
            Block.from_list(
                [
                    ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
                    ttnn.Tensor(torch.full(TILE_SHAPE, 10.0)),
                ],
                shape=(2, 1),
            )
        )

    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(2, 1), buffer_factor=4)
    with dfb.wait() as read_block:
        # Use waited block as STORE_SRC before context exit
        out_block = out_dfb.reserve()
        out_block.store(read_block)
        out_block.push()

    # Test multiple iterations
    for i in range(3):
        with dfb.reserve() as write_block:
            write_block.store(
                Block.from_list(
                    [
                        ttnn.Tensor(torch.full(TILE_SHAPE, float(i * 2))),
                        ttnn.Tensor(torch.full(TILE_SHAPE, float(i * 2 + 1))),
                    ],
                    shape=(2, 1),
                )
            )

        with dfb.wait() as read_block:
            # Use waited block as STORE_SRC before context exit
            out_block = out_dfb.reserve()
            out_block.store(read_block)
            out_block.push()

            # Verify data correctness for this iteration
            read_data = read_block.to_list()
            assert read_data[0].to_torch()[0, 0].item() == float(i * 2)
            assert read_data[1].to_torch()[0, 0].item() == float(i * 2 + 1)

    print("Reserve-store-push-pop workflow test passed!")


def test_make_dataflow_buffer_like_basic() -> None:
    """Test make_dataflow_buffer_like with basic usage."""
    from python.sim import ttl

    x = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)

    assert isinstance(x_dfb, DataflowBuffer)
    assert x_dfb.shape == (1, 1)
    assert x_dfb.capacity_tiles == 2
    assert x_dfb.buffer_factor == 2

    # DFBs are always initialized; the full reserve/store/push cycle should succeed.
    blk = x_dfb.reserve()
    blk.store(Block.from_tensor(make_zeros_tile()))
    blk.push()

    print("make_dataflow_buffer_like basic test passed!")


def test_make_dataflow_buffer_like_infers_type() -> None:
    """Test that make_dataflow_buffer_like correctly infers the element type."""
    from python.sim import ttl

    tensor = make_rand_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

    dfb = ttl.make_dataflow_buffer_like(tensor, shape=(2, 2), buffer_factor=3)

    assert dfb.shape == (2, 2)
    assert dfb.capacity_tiles == 12
    assert dfb.buffer_factor == 3

    print("make_dataflow_buffer_like type inference test passed!")


def test_make_dataflow_buffer_like_multiple_tensors() -> None:
    """Test make_dataflow_buffer_like with multiple different tensors."""
    from python.sim import ttl

    a = make_rand_tensor(TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4)
    b = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1])
    c = make_ones_tensor(TILE_SHAPE[0], TILE_SHAPE[1] * 2)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(4, 4), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 2), buffer_factor=3)

    assert a_dfb.shape == (4, 4)
    assert a_dfb.capacity_tiles == 32

    assert b_dfb.shape == (2, 1)
    assert b_dfb.capacity_tiles == 4

    assert c_dfb.shape == (1, 2)
    assert c_dfb.capacity_tiles == 6

    print("make_dataflow_buffer_like multiple tensors test passed!")


def test_make_dataflow_buffer_like_with_example_pattern() -> None:
    """Test make_dataflow_buffer_like with realistic example pattern."""
    from python.sim import ttl

    a_in = make_rand_tensor(128, 128)
    b_in = make_rand_tensor(128, 128)
    out = make_zeros_tensor(128, 128)

    granularity = 4
    buffer_factor = 2

    a_dfb = ttl.make_dataflow_buffer_like(
        a_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(granularity, 1), buffer_factor=buffer_factor
    )

    for dfb in [a_dfb, b_dfb, out_dfb]:
        assert dfb.shape == (granularity, 1)
        assert dfb.capacity_tiles == granularity * buffer_factor

    print("make_dataflow_buffer_like example pattern test passed!")


def test_can_wait_and_can_reserve() -> None:
    """Test can_wait() and can_reserve() methods."""
    element = make_ones_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

    # Initially, buffer is empty
    # can_reserve should return True (we have 2 free tiles)
    assert dfb.can_reserve() is True
    # can_wait should return False (no data available)
    assert dfb.can_wait() is False

    # Reserve and push one tile
    block = dfb.reserve()
    block.store(Block.from_tensor(make_ones_tile()))
    block.push()

    # Now we have 1 tile visible, 1 tile free
    assert dfb.can_wait() is True  # 1 tile available to read
    assert dfb.can_reserve() is True  # 1 tile free to write

    # Reserve and push another tile (buffer now full)
    block = dfb.reserve()
    tile = ttnn.rand(TILE_SHAPE)
    tile.to_torch().fill_(2.0)
    block.store(Block.from_tensor(tile))
    block.push()

    # Now we have 2 tiles visible, 0 tiles free
    assert dfb.can_wait() is True  # Still have data to read
    assert dfb.can_reserve() is False  # No free space

    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    read1 = dfb.wait()

    # After wait(), we have 1 tile read-locked, 1 tile still visible, 0 tiles free
    assert dfb.can_wait() is True  # Can still wait for the second visible tile
    assert dfb.can_reserve() is False  # No free tiles (both occupied)

    # Pop the first tile - use waited block as STORE_SRC first
    out_block = out_dfb.reserve()
    out_block.store(read1)
    out_block.push()
    read1.pop()

    # Now we have 1 tile visible, 1 tile free
    assert dfb.can_wait() is True  # Still have 1 tile to read
    assert dfb.can_reserve() is True  # Have 1 free tile

    # Pop the last tile - use waited block as STORE_SRC first
    read2 = dfb.wait()
    out_block2 = out_dfb.reserve()
    out_block2.store(read2)
    out_block2.push()
    read2.pop()

    # Back to empty state
    assert dfb.can_wait() is False  # No data available
    assert dfb.can_reserve() is True  # All tiles free

    print("can_wait() and can_reserve() test passed!")


def test_can_methods_multi_tile() -> None:
    """Test can_wait() and can_reserve() with multi-tile operations."""
    element = make_element_for_buffer_shape((2, 1))
    dfb = DataflowBuffer(likeness_tensor=element, shape=(2, 1), buffer_factor=3)

    # Initially empty
    assert dfb.can_reserve() is True  # 6 free tiles, need 2
    assert dfb.can_wait() is False  # 0 visible tiles, need 2

    # Reserve and push 2 tiles
    block = dfb.reserve()
    tile0, tile1 = ttnn.rand(TILE_SHAPE), ttnn.rand(TILE_SHAPE)
    tile0.to_torch().fill_(1.0)
    tile1.to_torch().fill_(2.0)
    block.store(Block.from_list([tile0, tile1], shape=(2, 1)))
    block.push()

    # 2 visible, 4 free
    assert dfb.can_wait() is True  # Have 2 tiles
    assert dfb.can_reserve() is True  # Have 4 free

    # Reserve and push 2 more tiles
    block = dfb.reserve()
    tile0, tile1 = ttnn.rand(TILE_SHAPE), ttnn.rand(TILE_SHAPE)
    tile0.to_torch().fill_(3.0)
    tile1.to_torch().fill_(4.0)
    block.store(Block.from_list([tile0, tile1], shape=(2, 1)))
    block.push()

    # 4 visible, 2 free
    assert dfb.can_wait() is True  # Have 4 tiles
    assert dfb.can_reserve() is True  # Have 2 free (exactly what we need)

    # Reserve and push 2 more tiles (buffer full)
    block = dfb.reserve()
    tile0, tile1 = ttnn.rand(TILE_SHAPE), ttnn.rand(TILE_SHAPE)
    tile0.to_torch().fill_(5.0)
    tile1.to_torch().fill_(6.0)
    block.store(Block.from_list([tile0, tile1], shape=(2, 1)))
    block.push()

    # 6 visible, 0 free
    assert dfb.can_wait() is True  # Have 6 tiles
    assert dfb.can_reserve() is False  # Have 0 free (need 2)

    print("can_wait() and can_reserve() multi-tile test passed!")


def test_can_methods_always_work() -> None:
    """Test that can_wait() and can_reserve() work on freshly created DFBs."""
    from python.sim import ttl

    x = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)
    dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)

    # DFBs are always initialized; both methods should return valid results.
    assert dfb.can_wait() is False  # empty buffer
    assert dfb.can_reserve() is True  # free space available

    print("can_wait() and can_reserve() always-initialized test passed!")


def test_context_manager_syntax() -> None:
    """Test the context manager (with statement) syntax for reserve and wait."""
    element = make_ones_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

    # Test reserve with context manager
    test_data = make_ones_tile()
    with dfb.reserve() as write_view:
        write_view.store(Block.from_tensor(test_data))
        # push() is automatically called on exit

    # Test wait with context manager
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    with dfb.wait() as read_view:
        out_block = out_dfb.reserve()
        out_block.store(read_view)
        out_block.push()

    write_view2 = dfb.reserve()
    write_view2.store(Block.from_tensor(make_zeros_tile()))
    write_view2.push()

    read_view2 = dfb.wait()
    out_block2 = out_dfb.reserve()
    out_block2.store(read_view2)
    out_block2.push()
    read_view2.pop()

    dfb.reset()

    with dfb.reserve() as w1:
        w1.store(Block.from_tensor(make_ones_tile()))

    dfb2 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    with dfb2.reserve() as w2:
        w2.store(Block.from_tensor(make_zeros_tile()))

    out_dfb3 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    out_dfb4 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    with dfb.wait() as r1, dfb2.wait() as r2:
        # Use waited blocks as STORE_SRC before context managers exit and call pop
        out_block3 = out_dfb3.reserve()
        out_block3.store(r1)
        out_block3.push()
        out_block4 = out_dfb4.reserve()
        out_block4.store(r2)
        out_block4.push()

        # Verify data correctness
        d1 = r1.to_list()[0]
        d2 = r2.to_list()[0]
        assert d1 is not None
        assert d2 is not None
        # Verify shape and type
        assert d1.to_torch().shape == (32, 32)
        assert d2.to_torch().shape == (32, 32)

    print("Context manager syntax test passed!")


def test_store_accumulate_first_assigns() -> None:
    """Test that the first store(acc=True) assigns instead of accumulates."""
    element = make_element_for_buffer_shape((3, 1))
    dfb = DataflowBuffer(likeness_tensor=element, shape=(3, 1), buffer_factor=2)

    with dfb.reserve() as block:
        import torch
        from python.sim import ttnn, TILE_SHAPE

        values1 = Block.from_list(
            [
                ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
                ttnn.Tensor(torch.full(TILE_SHAPE, 10.0)),
                ttnn.Tensor(torch.full(TILE_SHAPE, 15.0)),
            ],
            shape=(3, 1),
        )

        # First store(acc=True) - should assign (y = x), not accumulate (y += x)
        block.store(values1, acc=True)

        values2 = Block.from_list(
            [
                ttnn.Tensor(torch.full(TILE_SHAPE, 3.0)),
                ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
                ttnn.Tensor(torch.full(TILE_SHAPE, 9.0)),
            ],
            shape=(3, 1),
        )

        # Second store(acc=True) - should accumulate (y += x)
        block.store(values2, acc=True)

        # Verify results using to_list()
        result = block.to_list()
        assert result[0].to_torch()[0, 0].item() == 8.0  # 5 + 3
        assert result[1].to_torch()[0, 0].item() == 16.0  # 10 + 6
        assert result[2].to_torch()[0, 0].item() == 24.0  # 15 + 9

    print("Store accumulate first assigns test passed!")


def test_store_accumulate_vs_regular_store() -> None:
    """Test that regular store() and store(acc=True) have different paths."""
    element = make_element_for_buffer_shape((2, 1))
    dfb = DataflowBuffer(likeness_tensor=element, shape=(2, 1), buffer_factor=2)

    import torch
    from python.sim import ttnn, TILE_SHAPE

    # Test 1: Regular store() followed by push (cannot use store(acc=True) after)
    with dfb.reserve() as block1:
        block1.store(
            Block.from_list(
                [
                    ttnn.Tensor(torch.full(TILE_SHAPE, 7.0)),
                    ttnn.Tensor(torch.full(TILE_SHAPE, 14.0)),
                ],
                shape=(2, 1),
            )
        )  # Regular store

    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(2, 1), buffer_factor=2)
    with dfb.wait() as block_read:
        # Use waited block as STORE_SRC before context exit
        out_block = out_dfb.reserve()
        out_block.store(block_read)
        out_block.push()

    # Test 2: store(acc=True) path - can be called multiple times
    with dfb.reserve() as block2:
        block2.store(
            Block.from_list(
                [
                    ttnn.Tensor(torch.full(TILE_SHAPE, 2.0)),
                    ttnn.Tensor(torch.full(TILE_SHAPE, 4.0)),
                ],
                shape=(2, 1),
            ),
            acc=True,
        )  # First: assigns

        block2.store(
            Block.from_list(
                [
                    ttnn.Tensor(torch.full(TILE_SHAPE, 3.0)),
                    ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
                ],
                shape=(2, 1),
            ),
            acc=True,
        )  # Second: accumulates

        # Verify accumulation worked: 2+3=5, 4+6=10
        result = block2.to_list()
        assert result[0].to_torch()[0, 0].item() == 5.0
        assert result[1].to_torch()[0, 0].item() == 10.0

    print("Store accumulate vs regular store test passed!")


# ---------------------------------------------------------------------------
# Ring-buffer operation tests
# ---------------------------------------------------------------------------


def test_dataflow_buffer_basic_flow(configured_dfb8: DataflowBuffer) -> None:
    """Test ring-buffer mechanics via the public API at operation granularity."""
    element = make_ones_tile()
    in_dfb = configured_dfb8
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=8)

    stats = in_dfb.stats()
    assert stats.capacity == 8
    assert stats.visible == 0
    assert stats.free == 8

    # Push 4 operations one by one, verify stats update correctly.
    test_tensors = [make_full_tensor(32, 32, i + 1.0) for i in range(4)]
    for t in test_tensors:
        blk = in_dfb.reserve()
        blk.store(Block.from_tensor(t))
        blk.push()

    stats = in_dfb.stats()
    assert stats.visible == 4
    assert stats.free == 4

    # Consume 4 operations: route through out_dfb to satisfy the COMPUTE state machine.
    for i in range(4):
        read_blk = in_dfb.wait()
        out_blk = out_dfb.reserve()
        out_blk.store(read_blk)
        out_blk.push()
        read_blk.pop()

        stored = out_dfb.wait()
        values = stored.to_list()
        assert len(values) == 1
        assert tensors_exact_equal(values[0], test_tensors[i])
        drain_blk = DataflowBuffer(
            likeness_tensor=element, shape=(1, 1), buffer_factor=2
        )
        drain = drain_blk.reserve()
        drain.store(stored)
        drain.push()
        stored.pop()

    stats = in_dfb.stats()
    assert stats.visible == 0

    # Fill buffer completely (8 operations) then drain, exercising ring wraparound.
    test_tensors2 = [make_full_tensor(32, 32, float(i)) for i in range(8)]
    for t in test_tensors2:
        blk = in_dfb.reserve()
        blk.store(Block.from_tensor(t))
        blk.push()

    stats = in_dfb.stats()
    assert stats.visible == 8
    assert stats.free == 0

    out_dfb2 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=8)
    for i in range(8):
        read_blk = in_dfb.wait()
        out_blk = out_dfb2.reserve()
        out_blk.store(read_blk)
        out_blk.push()
        read_blk.pop()

    stats = in_dfb.stats()
    assert stats.visible == 0


def test_dfb_pages_nonblocking(configured_dfb8: DataflowBuffer) -> None:
    """Test that stats() reflects operation-granularity state at each step."""
    element = make_ones_tile()
    dfb = configured_dfb8
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

    assert dfb.stats().visible == 0
    assert dfb.stats().free == 8

    # After reserve: one slot is reserved, free decreases.
    blk = dfb.reserve()
    assert dfb.stats().reserved == 1
    assert dfb.stats().free == 7

    # After push: reserved becomes visible.
    blk.store(Block.from_tensor(make_full_tensor(32, 32, 1.0)))
    blk.push()
    assert dfb.stats().visible == 1
    assert dfb.stats().reserved == 0

    # After wait+pop: visible decreases (route through out_dfb per COMPUTE state machine).
    read_blk = dfb.wait()
    out_blk = out_dfb.reserve()
    out_blk.store(read_blk)
    out_blk.push()
    read_blk.pop()
    assert dfb.stats().visible == 0


def test_per_core_dfb_limit_exceeds_max() -> None:
    """Test that Program raises when a kernel's DataflowBuffer count exceeds the configured limit."""
    from python.sim.program import get_max_dfbs, Program
    from python.sim import ttl

    max_dfbs = get_max_dfbs()
    assert max_dfbs == 32

    element = make_ones_tile()

    @ttl.compute()
    def noop_compute():
        pass

    @ttl.datamovement()
    def noop_dm():
        pass

    prog = Program(noop_compute, noop_dm, noop_dm, grid=(1,))

    # Inject max_dfbs + 1 DataflowBuffers into the context.
    for i in range(max_dfbs + 1):
        prog.context[f"dfb_{i}"] = DataflowBuffer(likeness_tensor=element, shape=(1, 1))

    with pytest.raises(RuntimeError, match="exceeds the hardware limit"):
        prog()


def test_heterogeneous_dfbs_independent() -> None:
    """Test that multiple DataflowBuffers operate independently."""
    set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_full_tensor(64, 64, 1.0)

        dfb1 = DataflowBuffer(likeness_tensor=element, shape=(2, 2), buffer_factor=2)
        dfb2 = DataflowBuffer(likeness_tensor=element, shape=(2, 2), buffer_factor=2)

        write1 = dfb1.reserve()
        write1.store(
            Block.from_list(
                [make_full_tensor(32, 32, i + 1.0) for i in range(len(write1))],
                shape=(2, 2),
            )
        )
        write1.push()

        read1 = dfb1.wait()
        write1_2 = dfb1.reserve()
        write1_2.store(read1)
        read1.pop()
        write1_2.push()

        write2 = dfb2.reserve()
        write2.store(
            Block.from_list(
                [make_full_tensor(32, 32, i + 10.0) for i in range(len(write2))],
                shape=(2, 2),
            )
        )
        write2.push()

        read2 = dfb2.wait()
        write2_2 = dfb2.reserve()
        write2_2.store(read2)
        read2.pop()
        write2_2.push()
    finally:
        set_current_thread_type(None)


def test_two_dfbs_independent_state() -> None:
    """Test that two DataflowBuffers have fully independent ring-buffer state."""
    set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_full_tensor(32, 32, 1.0)

        dfb1 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
        dfb2 = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

        write1 = dfb1.reserve()
        write1.store(Block.from_tensor(make_full_tensor(32, 32, 42.0)))
        write1.push()

        write2 = dfb2.reserve()
        write2.store(Block.from_tensor(make_full_tensor(32, 32, 0.0)))
        write2.push()

        read1 = dfb1.wait()
        write1_2 = dfb1.reserve()
        write1_2.store(read1)
        read1.pop()
        write1_2.push()

        read2 = dfb2.wait()
        write2_2 = dfb2.reserve()
        write2_2.store(read2)
        read2.pop()
        write2_2.push()
    finally:
        set_current_thread_type(None)


# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------


def test_matmul_1x4_times_4x1_shape():
    """Test matmul with (1,4) @ (4,1) produces (1,1) result, not (4,4).

    This is a regression test for a bug where Block.__matmul__ was using
    broadcasting logic (result_shape = (max(1,4), max(4,1)) = (4,4))
    instead of matmul logic (result_shape = (1,1)).
    """
    a_block = Block.from_list(
        [Tensor(torch.ones((32, 32))) for _ in range(4)], shape=(1, 4)
    )
    b_block = Block.from_list(
        [Tensor(torch.ones((32, 32))) for _ in range(4)], shape=(4, 1)
    )
    result = a_block @ b_block
    assert result.shape == (1, 1), f"Expected (1,1) but got {result.shape}"
    assert (
        len(result.to_list()) == 1
    ), f"Expected 1 tile but got {len(result.to_list())}"


def test_matmul_2x3_times_3x2_shape():
    """Test matmul with (2,3) @ (3,2) produces (2,2) result."""
    a_block = Block.from_list(
        [Tensor(torch.ones((32, 32))) for _ in range(6)], shape=(2, 3)
    )
    b_block = Block.from_list(
        [Tensor(torch.ones((32, 32))) for _ in range(6)], shape=(3, 2)
    )
    result = a_block @ b_block
    assert result.shape == (2, 2), f"Expected (2,2) but got {result.shape}"
    assert (
        len(result.to_list()) == 4
    ), f"Expected 4 tiles but got {len(result.to_list())}"


def test_matmul_1x1_times_1x4_shape():
    """Test matmul with (1,1) @ (1,4) produces (1,4) result."""
    a_block = Block.from_list([Tensor(torch.ones((32, 32)))], shape=(1, 1))
    b_block = Block.from_list(
        [Tensor(torch.ones((32, 32))) for _ in range(4)], shape=(1, 4)
    )
    result = a_block @ b_block
    assert result.shape == (1, 4), f"Expected (1,4) but got {result.shape}"
    assert (
        len(result.to_list()) == 4
    ), f"Expected 4 tiles but got {len(result.to_list())}"


def test_matmul_4x1_times_1x4_shape():
    """Test matmul with (4,1) @ (1,4) produces (4,4) result."""
    a_block = Block.from_list(
        [Tensor(torch.ones((32, 32))) for _ in range(4)], shape=(4, 1)
    )
    b_block = Block.from_list(
        [Tensor(torch.ones((32, 32))) for _ in range(4)], shape=(1, 4)
    )
    result = a_block @ b_block
    assert result.shape == (4, 4), f"Expected (4,4) but got {result.shape}"
    assert (
        len(result.to_list()) == 16
    ), f"Expected 16 tiles but got {len(result.to_list())}"


def test_matmul_1x4_times_4x1_values():
    """Test matmul correctness for (1,4) @ (4,1) grid."""
    a_block = Block.from_list(
        [Tensor(torch.full((32, 32), float(i + 1))) for i in range(4)], shape=(1, 4)
    )
    b_block = Block.from_list(
        [Tensor(torch.full((32, 32), float(i + 1))) for i in range(4)], shape=(4, 1)
    )
    result = a_block @ b_block
    assert result.shape == (1, 1)

    # Each matmul tile: sum(i^2 * 32 for i in [1,2,3,4]) = 32 * 30 = 960
    result_tensor = result.to_list()[0].to_torch()
    expected_value = 32.0 * (1 * 1 + 2 * 2 + 3 * 3 + 4 * 4)
    assert torch.allclose(
        result_tensor, torch.full((32, 32), expected_value)
    ), f"Expected all values to be {expected_value}, got {result_tensor[0, 0]}"


# ---------------------------------------------------------------------------
# 1-D tensor support tests
# ---------------------------------------------------------------------------


def test_1d_tile_count_from_tensor():
    """tile_count_from_tensor correctly counts tiles in a 1-D tensor."""
    from python.sim.dfb import tile_count_from_tensor

    assert tile_count_from_tensor(Tensor(torch.zeros(32))) == 1
    assert tile_count_from_tensor(Tensor(torch.zeros(64))) == 2
    assert tile_count_from_tensor(Tensor(torch.zeros(128))) == 4
    # Degenerate: size==1 counts as 1 tile
    assert tile_count_from_tensor(Tensor(torch.zeros(1))) == 1


def test_1d_block_from_tensor():
    """Block.from_tensor infers shape (TK,) for a 1-D tensor."""
    t = Tensor(torch.zeros(64))
    b = Block.from_tensor(t)
    assert b.shape == (2,), f"Expected shape (2,) but got {b.shape}"
    assert len(b) == 2


def test_1d_block_to_list():
    """Block.to_list splits a 1-D block into individual tile vectors."""
    data = torch.arange(64, dtype=torch.float32)
    b = Block.from_tensor(Tensor(data))
    tiles = b.to_list()
    assert len(tiles) == 2
    assert tiles[0].to_torch().shape == (32,)
    assert tiles[1].to_torch().shape == (32,)
    assert torch.allclose(tiles[0].to_torch(), data[:32])
    assert torch.allclose(tiles[1].to_torch(), data[32:])


def test_1d_block_from_list():
    """Block.from_list assembles 1-D tile vectors into a single 1-D Block."""
    t0 = Tensor(torch.ones(32) * 1.0)
    t1 = Tensor(torch.ones(32) * 2.0)
    b = Block.from_list([t0, t1], shape=(2,))
    assert b.shape == (2,)
    raw = b.to_tensor().to_torch()
    assert raw.shape == (64,)
    assert torch.allclose(raw[:32], torch.ones(32))
    assert torch.allclose(raw[32:], torch.ones(32) * 2.0)


def test_1d_dataflow_buffer_reserve_push_wait_pop():
    """DataflowBuffer with 1-D shape correctly reserves, pushes, and delivers data."""
    from python.sim.blockstate import ThreadType
    from python.sim.context import set_current_thread_type, clear_current_thread_type

    element = Tensor(torch.zeros(32))
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1,), buffer_factor=2)

    assert dfb.shape == (1,)
    assert dfb.capacity_tiles == 2

    set_current_thread_type(ThreadType.COMPUTE)
    try:
        write = dfb.reserve()
        assert len(write) == 1

        data = Tensor(torch.arange(32, dtype=torch.float32))
        write.store(Block.from_tensor(data))
        write.push()

        read = dfb.wait()
        assert len(read) == 1
        result = read.to_list()
        assert len(result) == 1
        assert result[0].to_torch().shape == (32,)
        assert torch.allclose(result[0].to_torch(), data.to_torch())

        out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1,), buffer_factor=2)
        out_block = out_dfb.reserve()
        out_block.store(read)
        out_block.push()
        read.pop()
    finally:
        clear_current_thread_type()


def test_1d_multi_tile_dataflow_buffer():
    """DataflowBuffer with 1-D shape (4,) operates over 4 tiles per operation."""
    from python.sim.blockstate import ThreadType
    from python.sim.context import set_current_thread_type, clear_current_thread_type

    # Full buffer element shape for 4 tiles of size 32 each
    element = Tensor(torch.zeros(128))
    dfb = DataflowBuffer(likeness_tensor=element, shape=(4,), buffer_factor=2)

    assert dfb.shape == (4,)
    assert dfb.capacity_tiles == 8

    set_current_thread_type(ThreadType.COMPUTE)
    try:
        write = dfb.reserve()
        assert len(write) == 4
        # Element tensor should be (128,) = 4 * 32
        assert write.to_tensor().to_torch().shape == (128,)

        tiles = [Tensor(torch.full((32,), float(i))) for i in range(4)]
        write.store(Block.from_list(tiles, shape=(4,)))
        write.push()

        read = dfb.wait()
        result = read.to_list()
        assert len(result) == 4
        for i, tile in enumerate(result):
            assert torch.allclose(tile.to_torch(), torch.full((32,), float(i)))

        out_dfb = DataflowBuffer(likeness_tensor=element, shape=(4,), buffer_factor=2)
        out_block = out_dfb.reserve()
        out_block.store(read)
        out_block.push()
        read.pop()
    finally:
        clear_current_thread_type()


def test_1d_tensor_tile_aligned_validation():
    """1-D tensors that are not tile-aligned (or size 1) are rejected by from_tensor."""
    from python.sim.dfb import Block

    # Aligned: 32, 64, 1
    Block.from_tensor(Tensor(torch.zeros(32)))
    Block.from_tensor(Tensor(torch.zeros(64)))
    Block.from_tensor(Tensor(torch.zeros(1)))

    # Not aligned
    with pytest.raises(ValueError, match="multiple of TILE_SHAPE"):
        Block.from_tensor(Tensor(torch.zeros(33)))

    with pytest.raises(ValueError, match="multiple of TILE_SHAPE"):
        Block.from_tensor(Tensor(torch.zeros(16)))


def test_1d_arithmetic_on_blocks():
    """Basic arithmetic on 1-D blocks works element-wise."""
    a = Block.from_tensor(Tensor(torch.ones(64) * 2.0))
    b = Block.from_tensor(Tensor(torch.ones(64) * 3.0))
    c = a + b
    assert c.shape == (2,)
    assert torch.allclose(c.to_tensor().to_torch(), torch.ones(64) * 5.0)

    d = a * b
    assert torch.allclose(d.to_tensor().to_torch(), torch.ones(64) * 6.0)


# ---------------------------------------------------------------------------
# High-dimensional (4-D, 5-D, 6-D) grid round-trip tests
#
# These verify that Block.from_list and to_list are inverses for grids with
# nb >= 2 batch dimensions (nb = ndim - 2), exercising the general permute
# formula in from_list.
# ---------------------------------------------------------------------------


def _make_grid(shape):
    """Return a list of distinct (32x32) tiles for a tile grid of given shape."""
    n = 1
    for d in shape:
        n *= d
    return [Tensor(torch.full((32, 32), float(i))) for i in range(n)]


def _round_trip(shape):
    """Build a Block from a tile list, split it back, and verify values."""
    tiles = _make_grid(shape)
    block = Block.from_list(tiles, shape=shape)
    assert block.shape == shape
    recovered = block.to_list()
    assert len(recovered) == len(tiles)
    for i, (orig, rec) in enumerate(zip(tiles, recovered)):
        assert torch.allclose(
            orig.to_torch(), rec.to_torch()
        ), f"Tile {i} mismatch for shape {shape}"


def test_4d_grid_round_trip():
    """Block.from_list / to_list round-trip for a 4-D tile grid (2 batch dims)."""
    # shape (2, 3, 2, 2): nb=2 batch dims, TM=2, TK=2 -> 2*3*2*2 = 24 tiles
    _round_trip((2, 3, 2, 2))


def test_5d_grid_round_trip():
    """Block.from_list / to_list round-trip for a 5-D tile grid (3 batch dims)."""
    # shape (2, 2, 2, 2, 2): nb=3, TM=2, TK=2 -> 32 tiles
    _round_trip((2, 2, 2, 2, 2))


def test_6d_grid_round_trip():
    """Block.from_list / to_list round-trip for a 6-D tile grid (4 batch dims)."""
    # shape (2, 2, 2, 2, 2, 2): nb=4, TM=2, TK=2 -> 64 tiles
    _round_trip((2, 2, 2, 2, 2, 2))


def test_4d_grid_values_are_distinct():
    """Each tile in a 4-D block retains its identity value after a round-trip."""
    shape = (3, 2, 2, 2)  # 24 tiles
    tiles = _make_grid(shape)
    block = Block.from_list(tiles, shape=shape)
    recovered = block.to_list()
    for i, rec in enumerate(recovered):
        expected_val = float(i)
        assert torch.all(
            rec.to_torch() == expected_val
        ), f"Tile {i}: expected all {expected_val}, got {rec.to_torch()}"


def test_4d_grid_backing_tensor_shape():
    """The backing tensor of a 4-D block has the correct element-space shape."""
    shape = (2, 3, 2, 4)  # nb=2, TM=2, TK=4; backing shape = (2,3, 64, 128)
    tiles = _make_grid(shape)
    block = Block.from_list(tiles, shape=shape)
    raw = block.to_tensor().to_torch()
    assert raw.shape == (2, 3, 64, 128), f"Unexpected shape {raw.shape}"


def test_5d_grid_from_tensor_infers_shape():
    """Block.from_tensor correctly infers a 5-D tile-grid shape."""
    # 5-D element tensor: (2, 3, 4, 64, 96) -> grid (2, 3, 4, 2, 3)
    data = torch.zeros(2, 3, 4, 64, 96)
    block = Block.from_tensor(Tensor(data))
    assert block.shape == (2, 3, 4, 2, 3)


def test_store_broadcast_expansion():
    """Test that store() expands source blocks to match destination shape when using broadcast()."""
    # Create DFBs for testing broadcast expansion with element-based semantics

    # Test 1: Broadcast (1, 2) -> (3, 2) with degenerate tiles (element size 1 in first dim)
    # Source has full buffer shape (1, 64) - 1 row, 2×32=64 columns
    # First dim (outermost/rows) has element size 1, broadcast along dim 0 (outermost)
    src_elem = Tensor(
        torch.zeros(1, 64, dtype=torch.bfloat16)
    )  # Full buffer element shape
    src_dfb = DataflowBuffer(likeness_tensor=src_elem, shape=(1, 2), buffer_factor=2)

    # Destination has full buffer shape (96, 64) - 3×32=96 rows, 2×32=64 columns
    dst_elem = Tensor(
        torch.zeros(96, 64, dtype=torch.bfloat16)
    )  # Full buffer element shape
    dst_dfb = DataflowBuffer(likeness_tensor=dst_elem, shape=(3, 2), buffer_factor=2)

    with src_dfb.reserve() as src_blk:
        # Fill source with degenerate tiles: first column = 10, second column = 20
        src_tiles = [
            Tensor(torch.full((1, 32), 10.0, dtype=torch.bfloat16)),
            Tensor(torch.full((1, 32), 20.0, dtype=torch.bfloat16)),
        ]
        src_temp = Block.from_list(src_tiles, shape=(1, 2))
        src_blk.store(src_temp)

    with dst_dfb.reserve() as dst_blk:
        with src_dfb.wait() as src_wait:
            # Explicitly broadcast to expand row dimension (dims=[0] = outermost in standard Python indexing)
            broadcast_src = broadcast(src_wait, dims=[0])
            # Store with broadcast expansion
            dst_blk.store(broadcast_src)

            # Check that tiles were replicated correctly
            result = dst_blk.to_list()
            assert len(result) == 6
            # Each row should have the same values: [10, 20]
            for i in range(3):  # 3 rows
                assert torch.allclose(
                    result[i * 2].to_torch(),
                    torch.full((32, 32), 10.0, dtype=torch.bfloat16),
                )
                assert torch.allclose(
                    result[i * 2 + 1].to_torch(),
                    torch.full((32, 32), 20.0, dtype=torch.bfloat16),
                )

    # Test 2: Broadcast (1, 1) -> (2, 2) with proper element shapes
    # Source has full buffer shape (1, 1) - broadcast along both dimensions
    src2_elem = Tensor(
        torch.zeros(1, 1, dtype=torch.bfloat16)
    )  # Full buffer element shape
    src2_dfb = DataflowBuffer(likeness_tensor=src2_elem, shape=(1, 1), buffer_factor=2)

    # Destination has full buffer shape (64, 64) - 2×32=64 rows, 2×32=64 columns
    dst2_elem = Tensor(
        torch.zeros(64, 64, dtype=torch.bfloat16)
    )  # Full buffer element shape
    dst2_dfb = DataflowBuffer(likeness_tensor=dst2_elem, shape=(2, 2), buffer_factor=2)

    with src2_dfb.reserve() as src2_blk:
        src2_tiles = [Tensor(torch.full((1, 1), 42.0, dtype=torch.bfloat16))]
        src2_temp = Block.from_list(src2_tiles, shape=(1, 1))
        src2_blk.store(src2_temp)

    with dst2_dfb.reserve() as dst2_blk:
        with src2_dfb.wait() as src2_wait:
            # Explicitly broadcast to expand both dimensions (dims=[0, 1])
            broadcast_src2 = broadcast(src2_wait, dims=[0, 1])
            dst2_blk.store(broadcast_src2)

            # Check all tiles are 42.0
            result2 = dst2_blk.to_list()
            assert len(result2) == 4
            for tile in result2:
                assert torch.allclose(
                    tile.to_torch(), torch.full((32, 32), 42.0, dtype=torch.bfloat16)
                )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
