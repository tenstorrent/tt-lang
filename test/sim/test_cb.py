# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for DataflowBuffer and DFBAPI.

Covers the high-level DataflowBuffer interface (tensor-aware operations,
context manager syntax, state machine enforcement) and the low-level DFBAPI
(reserve/wait/push/pop primitives, threading, error contracts, DFB allocation).
"""

import threading
import time
from typing import List, Tuple

import pytest
import torch
from test_utils import (
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
from python.sim.dfb import (
    DFBAPI,
    Block,
    DataflowBuffer,
)
from python.sim.blockstate import (
    set_current_thread_type,
    clear_current_thread_type,
    ThreadType,
    BlockAcquisition,
)
from python.sim.ttnnsim import Tensor
from python.sim.dfbstate import DFBSlot
from python.sim.errors import DFBContractError, DFBTimeoutError
from python.sim.typedefs import DFBID


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
def api():
    """Provide a fresh DFBAPI instance for each test."""
    return DFBAPI()


@pytest.fixture
def configured_dfb(api: DFBAPI) -> Tuple[DFBAPI, DFBID]:
    """Create a configured DFB with capacity 4."""
    dfb_id = 0
    api.host_configure_dfb(dfb_id, 4, shape=(1, 1))
    return api, dfb_id


@pytest.fixture
def configured_dfb8(api: DFBAPI) -> Tuple[DFBAPI, DFBID]:
    """Create a configured DFB with capacity 8."""
    dfb_id = 0
    api.host_configure_dfb(dfb_id, 8, shape=(1, 1))
    return api, dfb_id


@pytest.fixture
def timeout_api() -> DFBAPI:
    """Create a DFBAPI instance with short timeout for timeout tests."""
    return DFBAPI(timeout=0.1)


def test_circular_buffer_basic(api: DFBAPI) -> None:
    """Test basic DataflowBuffer operations."""
    # Create a dataflow buffer for single tiles with buffer factor 2
    element = make_ones_tile()
    dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    # Verify basic properties
    assert dfb.shape == (1, 1)
    assert dfb.capacity_tiles == 2  # 1*1*2
    assert dfb.buffer_factor == 2

    # Test the buffer workflow
    # Producer: reserve -> write -> push
    write_view = dfb.reserve()
    assert len(write_view) == 1  # Should have space for 1 tile

    # Simulate writing data
    test_data = make_ones_tile()
    write_view.store([test_data])
    write_view.push()

    # Consumer: wait -> read -> pop
    read_view = dfb.wait()
    assert len(read_view) == 1  # Should have 1 tile available

    # Use waited block as source (STORE_SRC) before pop
    out_dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    out_block = out_dfb.reserve()
    out_block.store(read_view)
    out_block.push()

    # Verify data was transferred correctly
    read_data = read_view.to_list()
    assert read_data[0] is not None
    assert tensors_equal(read_data[0], test_data)

    read_view.pop()

    print("Basic DataflowBuffer test passed!")


def test_circular_buffer_multi_tile(api: DFBAPI) -> None:
    """Test DataflowBuffer with multiple tiles per operation."""
    # Create a dataflow buffer for 2x1 tiles (2 tiles per operation)
    element = make_ones_tile()
    dfb = DataflowBuffer(element=element, shape=(2, 1), buffer_factor=3, api=api)

    # Verify properties
    assert dfb.shape == (2, 1)
    assert dfb.capacity_tiles == 6  # 2*1*3

    # Test reserve/push
    write_view = dfb.reserve()
    assert len(write_view) == 2  # Should have space for 2 tiles

    # Fill with test data
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 1))
        tiles.append(tile)
    write_view.store(tiles)

    write_view.push()

    # Test wait/pop
    read_view = dfb.wait()
    assert len(read_view) == 2  # Should have 2 tiles available

    # Use waited block as source (STORE_SRC) before pop
    out_dfb = DataflowBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)
    out_block = out_dfb.reserve()
    out_block.store(read_view)
    out_block.push()

    # Verify data was transferred correctly
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


def test_copy_operations_with_dm_context(api: DFBAPI) -> None:
    """Test copy operations between tensor and DataflowBuffer with proper DM thread context.

    This replaces the old test_copy_operations that was disabled due to lack of thread context.
    """

    # Set DM thread context (required for copy operations)
    set_current_thread_type(ThreadType.DM)

    try:
        # Create test tensors
        tensor_a = make_rand_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)  # 2x2 tiles

        # Create dataflow buffer
        element = make_ones_tile()
        dfb_a = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

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


def test_error_handling(api: DFBAPI) -> None:
    """Test error conditions."""
    # Test invalid shape
    element = make_ones_tile()
    with pytest.raises(ValueError):
        DataflowBuffer(element=element, shape=(0, 1), api=api)  # Invalid shape

    with pytest.raises(ValueError):
        DataflowBuffer(element=element, shape=(1, 2, 3), api=api)  # type: ignore # Wrong shape dimensions

    # Test invalid buffer factor
    with pytest.raises(ValueError):
        DataflowBuffer(element=element, shape=(1, 1), buffer_factor=0, api=api)

    # Test operations without proper setup
    dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

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


def test_copy_in_dm_thread_context(api: DFBAPI) -> None:
    """Test copy operations with proper DM thread context.

    This test demonstrates the full workflow:
    - DM thread: copy data into DFBs (reserve + copy + push)
    - Switch to COMPUTE thread for consumption (wait + read + pop)
    """

    try:
        # Create tensors
        rows, cols = 128, 128
        granularity = 4

        a_in = make_rand_tensor(rows, cols)
        c_in = make_rand_tensor(TILE_SHAPE[0], cols)

        # Create circular buffers
        element = make_ones_tile()
        a_in_dfb = DataflowBuffer(
            element=element, shape=(granularity, 1), buffer_factor=2, api=api
        )
        c_in_dfb = DataflowBuffer(
            element=element, shape=(1, 1), buffer_factor=2, api=api
        )

        # Verify the circular buffers were created correctly
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

        # In COMPUTE thread, wait() blocks must be used as STORE_SRC before pop
        out_dfb = DataflowBuffer(
            element=element, shape=(1, 1), buffer_factor=2, api=api
        )
        out_block = out_dfb.reserve()
        out_block.store(c_data)
        out_block.push()
        c_data.pop()

        out_dfb2 = DataflowBuffer(
            element=element, shape=(granularity, 1), buffer_factor=2, api=api
        )
        out_block2 = out_dfb2.reserve()
        out_block2.store(a_data)
        out_block2.push()
        a_data.pop()

    finally:
        # Clean up thread context
        clear_current_thread_type()

    print("Copy in DM thread context test passed!")


def test_single_pending_reserve_constraint(api: DFBAPI) -> None:
    """Test that only one reserve() is allowed before push()."""
    from python.sim.copy import copy

    set_current_thread_type(ThreadType.DM)

    try:
        element = make_ones_tile()
        dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

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


def test_single_pending_wait_constraint(api: DFBAPI) -> None:
    """Test that only one wait() is allowed before pop()."""
    from python.sim.copy import copy

    set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_ones_tile()
        dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        # First populate the DFB with data (using DM thread)
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

        # After pop(), should be able to wait() again (if there's more data)
        # Use waited block as STORE_SRC before pop
        out_dfb = DataflowBuffer(
            element=element, shape=(1, 1), buffer_factor=2, api=api
        )
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


def test_reserve_store_push_pop_workflow(api: DFBAPI) -> None:
    """Test the complete reserve->store->push->wait->pop workflow.

    This tests the primary usage pattern for compute operations without
    using copy (which requires DM thread context).
    """
    import torch
    from python.sim import ttnn, TILE_SHAPE

    # Create dataflow buffer
    element = make_zeros_tile()
    dfb = DataflowBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)

    # Producer: reserve -> store -> push
    with dfb.reserve() as write_block:
        # Create test data
        data = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 10.0)),
        ]
        write_block.store(data)

    # Consumer: wait -> read -> pop
    out_dfb = DataflowBuffer(element=element, shape=(2, 1), buffer_factor=4, api=api)
    with dfb.wait() as read_block:
        # Use waited block as STORE_SRC before context exit
        out_block = out_dfb.reserve()
        out_block.store(read_block)
        out_block.push()

    # Test multiple iterations
    for i in range(3):
        with dfb.reserve() as write_block:
            data = [
                ttnn.Tensor(torch.full(TILE_SHAPE, float(i * 2))),
                ttnn.Tensor(torch.full(TILE_SHAPE, float(i * 2 + 1))),
            ]
            write_block.store(data)

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


def test_make_dataflow_buffer_like_basic(api: DFBAPI) -> None:
    """Test make_dataflow_buffer_like with basic usage."""
    from python.sim import ttl

    # Create a tensor
    x = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)

    # Create a dataflow buffer like x
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)

    # Verify it's a DataflowBuffer with correct properties
    assert isinstance(x_dfb, DataflowBuffer)
    assert x_dfb.shape == (1, 1)
    assert x_dfb.capacity_tiles == 2
    assert x_dfb.buffer_factor == 2

    # Verify it's not initialized (no API)
    assert x_dfb._api is None  # type: ignore[reportPrivateUsage]
    assert x_dfb._dfb_id is None  # type: ignore[reportPrivateUsage]

    # Verify that using it without initialization raises an error
    with pytest.raises(RuntimeError, match="not properly initialized"):
        x_dfb.reserve()

    print("make_dataflow_buffer_like basic test passed!")


def test_make_dataflow_buffer_like_infers_type(api: DFBAPI) -> None:
    """Test that make_dataflow_buffer_like correctly infers the element type."""
    from python.sim import ttl

    # Create a tensor
    tensor = make_rand_tensor(TILE_SHAPE[0], TILE_SHAPE[1])

    # Create a dataflow buffer like the tensor
    dfb = ttl.make_dataflow_buffer_like(tensor, shape=(2, 2), buffer_factor=3)

    # Verify properties
    assert dfb.shape == (2, 2)
    assert dfb.capacity_tiles == 12  # 2*2*3
    assert dfb.buffer_factor == 3

    # Verify it's not initialized
    assert dfb._api is None  # type: ignore[reportPrivateUsage]

    # Verify error when used without initialization
    with pytest.raises(RuntimeError, match="not properly initialized"):
        dfb.reserve()

    print("make_dataflow_buffer_like type inference test passed!")


def test_make_dataflow_buffer_like_multiple_tensors(api: DFBAPI) -> None:
    """Test make_dataflow_buffer_like with multiple different tensors."""
    from python.sim import ttl

    # Create different tensors
    a = make_rand_tensor(TILE_SHAPE[0] * 4, TILE_SHAPE[1] * 4)
    b = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)
    c = make_ones_tensor(TILE_SHAPE[0], TILE_SHAPE[1])

    # Create circular buffers for each
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 2), buffer_factor=3)

    # Verify all have correct properties
    assert a_dfb.shape == (1, 1)
    assert a_dfb.capacity_tiles == 2

    assert b_dfb.shape == (2, 1)
    assert b_dfb.capacity_tiles == 4  # 2*1*2

    assert c_dfb.shape == (1, 2)
    assert c_dfb.capacity_tiles == 6  # 1*2*3

    # Verify they're all uninitialized
    for dfb in [a_dfb, b_dfb, c_dfb]:
        assert dfb._api is None  # type: ignore[reportPrivateUsage]
        with pytest.raises(RuntimeError, match="not properly initialized"):
            dfb.reserve()

    print("make_dataflow_buffer_like multiple tensors test passed!")


def test_make_dataflow_buffer_like_with_example_pattern(api: DFBAPI) -> None:
    """Test make_dataflow_buffer_like with realistic example pattern."""
    from python.sim import ttl

    # Simulate example usage
    a_in = make_rand_tensor(128, 128)
    b_in = make_rand_tensor(128, 128)
    out = make_zeros_tensor(128, 128)

    granularity = 4
    buffer_factor = 2

    # Create circular buffers using make_dataflow_buffer_like
    a_dfb = ttl.make_dataflow_buffer_like(
        a_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    b_dfb = ttl.make_dataflow_buffer_like(
        b_in, shape=(granularity, 1), buffer_factor=buffer_factor
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(granularity, 1), buffer_factor=buffer_factor
    )

    # Verify all buffers have correct configuration
    for dfb in [a_dfb, b_dfb, out_dfb]:
        assert dfb.shape == (granularity, 1)
        assert dfb.capacity_tiles == granularity * buffer_factor
        # Verify they're uninitialized
        assert dfb._api is None  # type: ignore[reportPrivateUsage]

    # Verify that operations fail without initialization
    with pytest.raises(RuntimeError, match="not properly initialized"):
        a_dfb.reserve()

    print("make_dataflow_buffer_like example pattern test passed!")


def test_can_wait_and_can_reserve(api: DFBAPI) -> None:
    """Test can_wait() and can_reserve() methods."""
    # Create a dataflow buffer with buffer factor 2 (capacity = 2 tiles)
    element = make_ones_tile()
    dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    # Initially, buffer is empty
    # can_reserve should return True (we have 2 free tiles)
    assert dfb.can_reserve() is True
    # can_wait should return False (no data available)
    assert dfb.can_wait() is False

    # Reserve and push one tile
    block = dfb.reserve()
    block.store([make_ones_tile()])
    block.push()

    # Now we have 1 tile visible, 1 tile free
    assert dfb.can_wait() is True  # 1 tile available to read
    assert dfb.can_reserve() is True  # 1 tile free to write

    # Reserve and push another tile (buffer now full)
    block = dfb.reserve()
    tile = ttnn.rand(TILE_SHAPE)
    tile.to_torch().fill_(2.0)
    block.store([tile])
    block.push()

    # Now we have 2 tiles visible, 0 tiles free
    assert dfb.can_wait() is True  # Still have data to read
    assert dfb.can_reserve() is False  # No free space

    # Wait for the first tile
    out_dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
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


def test_can_methods_multi_tile(api: DFBAPI) -> None:
    """Test can_wait() and can_reserve() with multi-tile operations."""
    # Create a buffer that handles 2 tiles per operation, capacity = 6 tiles
    element = make_ones_tile()
    dfb = DataflowBuffer(element=element, shape=(2, 1), buffer_factor=3, api=api)

    # Initially empty
    assert dfb.can_reserve() is True  # 6 free tiles, need 2
    assert dfb.can_wait() is False  # 0 visible tiles, need 2

    # Reserve and push 2 tiles
    block = dfb.reserve()
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 1))
        tiles.append(tile)
    block.store(tiles)
    block.push()

    # 2 visible, 4 free
    assert dfb.can_wait() is True  # Have 2 tiles
    assert dfb.can_reserve() is True  # Have 4 free

    # Reserve and push 2 more tiles
    block = dfb.reserve()
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 3))
        tiles.append(tile)
    block.store(tiles)
    block.push()

    # 4 visible, 2 free
    assert dfb.can_wait() is True  # Have 4 tiles
    assert dfb.can_reserve() is True  # Have 2 free (exactly what we need)

    # Reserve and push 2 more tiles (buffer full)
    block = dfb.reserve()
    tiles = []
    for i in range(2):
        tile = ttnn.rand(TILE_SHAPE)
        tile.to_torch().fill_(float(i + 5))
        tiles.append(tile)
    block.store(tiles)
    block.push()

    # 6 visible, 0 free
    assert dfb.can_wait() is True  # Have 6 tiles
    assert dfb.can_reserve() is False  # Have 0 free (need 2)

    print("can_wait() and can_reserve() multi-tile test passed!")


def test_can_methods_uninitialized(api: DFBAPI) -> None:
    """Test that can_wait() and can_reserve() fail on uninitialized DFBs."""
    from python.sim import ttl

    x = make_zeros_tensor(TILE_SHAPE[0] * 2, TILE_SHAPE[1] * 2)
    dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), buffer_factor=2)

    # Both methods should raise RuntimeError on uninitialized DFB
    with pytest.raises(RuntimeError, match="not properly initialized"):
        dfb.can_wait()

    with pytest.raises(RuntimeError, match="not properly initialized"):
        dfb.can_reserve()

    print("can_wait() and can_reserve() uninitialized test passed!")


def test_context_manager_syntax(api: DFBAPI) -> None:
    """Test the context manager (with statement) syntax for reserve and wait."""
    # Create a dataflow buffer
    element = make_ones_tile()
    dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    # Test reserve with context manager
    test_data = make_ones_tile()
    with dfb.reserve() as write_view:
        write_view.store([test_data])
        # push() is automatically called on exit

    # Test wait with context manager
    out_dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    with dfb.wait() as read_view:
        # Use waited block as STORE_SRC before pop() is automatically called on exit
        out_block = out_dfb.reserve()
        out_block.store(read_view)
        out_block.push()
        # pop() is automatically called on exit

    # Verify that we can still use the old style (backward compatibility)
    write_view2 = dfb.reserve()
    write_view2.store([make_zeros_tile()])
    write_view2.push()

    read_view2 = dfb.wait()
    # Use waited block as STORE_SRC before pop
    out_block2 = out_dfb.reserve()
    out_block2.store(read_view2)
    out_block2.push()
    read_view2.pop()

    # Test with multiple context managers on same line
    dfb.reset()  # Reset to clean state

    # Write data first
    with dfb.reserve() as w1:
        w1.store([make_ones_tile()])

    # Create another DFB for multi-context test
    dfb2 = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    with dfb2.reserve() as w2:
        w2.store([make_zeros_tile()])

    # Test multiple wait contexts (simulating the matmul pattern)
    out_dfb3 = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    out_dfb4 = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
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


def test_store_accumulate_first_assigns(api: DFBAPI) -> None:
    """Test that the first store(acc=True) assigns instead of accumulates."""
    element = make_zeros_tile()
    dfb = DataflowBuffer(element=element, shape=(3, 1), buffer_factor=2, api=api)

    with dfb.reserve() as block:
        # Create test values
        import torch
        from python.sim import ttnn, TILE_SHAPE

        values1 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 10.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 15.0)),
        ]

        # First store(acc=True) - should assign (y = x), not accumulate (y += x)
        block.store(values1, acc=True)

        # Second store(acc=True) - should accumulate (y += x)
        values2 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 3.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 9.0)),
        ]
        block.store(values2, acc=True)

        # Verify results using to_list()
        result = block.to_list()
        assert result[0].to_torch()[0, 0].item() == 8.0  # 5 + 3
        assert result[1].to_torch()[0, 0].item() == 16.0  # 10 + 6
        assert result[2].to_torch()[0, 0].item() == 24.0  # 15 + 9

    print("Store accumulate first assigns test passed!")


def test_store_accumulate_vs_regular_store(api: DFBAPI) -> None:
    """Test that regular store() and store(acc=True) have different paths."""
    element = make_zeros_tile()
    dfb = DataflowBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)

    import torch
    from python.sim import ttnn, TILE_SHAPE

    # Test 1: Regular store() followed by push (cannot use store(acc=True) after)
    with dfb.reserve() as block1:
        values = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 7.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 14.0)),
        ]
        block1.store(values)  # Regular store

    # Verify we can read it back
    out_dfb = DataflowBuffer(element=element, shape=(2, 1), buffer_factor=2, api=api)
    with dfb.wait() as block_read:
        # Use waited block as STORE_SRC before context exit
        out_block = out_dfb.reserve()
        out_block.store(block_read)
        out_block.push()

    # Test 2: store(acc=True) path - can be called multiple times
    with dfb.reserve() as block2:
        values1 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 2.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 4.0)),
        ]
        block2.store(values1, acc=True)  # First: assigns

        values2 = [
            ttnn.Tensor(torch.full(TILE_SHAPE, 3.0)),
            ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
        ]
        block2.store(values2, acc=True)  # Second: accumulates

        # Verify accumulation worked: 2+3=5, 4+6=10
        result = block2.to_list()
        assert result[0].to_torch()[0, 0].item() == 5.0
        assert result[1].to_torch()[0, 0].item() == 10.0

    print("Store accumulate vs regular store test passed!")


def test_block_state_machine_restrictions(api: DFBAPI) -> None:
    """Test that block state machine enforces access restrictions."""
    element = make_zeros_tile()
    dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

    import torch
    from python.sim import ttnn, TILE_SHAPE

    # Test: Cannot index blocks - block indexing is not allowed
    block = dfb.reserve()

    # Attempting to index block should fail
    with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
        _ = block[0]

    # Store makes it RO (for regular store) or RW (for acc store)
    values = [ttnn.Tensor(torch.full(TILE_SHAPE, 5.0))]
    block.store(values, acc=True)

    block.push()

    # Test: Cannot write to RO (Read-Only) state after wait()
    read_block = dfb.wait()

    # Cannot write - wait() blocks expect STORE_SRC, not STORE
    with pytest.raises(RuntimeError, match="Cannot perform store.*Expected one of"):
        read_block.store([ttnn.Tensor(torch.full(TILE_SHAPE, 10.0))])

    # Use waited block as STORE_SRC before pop
    out_dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
    out_block = out_dfb.reserve()
    out_block.store(read_block)
    out_block.push()
    read_block.pop()

    print("Block state machine restrictions test passed!")


def test_copy_sets_block_to_na_state(api: DFBAPI) -> None:
    """Test that copy operations set blocks to NA (No Access) state."""
    from python.sim.typedefs import Span
    import torch
    from python.sim import ttnn

    # Set thread type to DM (required for copy operations)
    set_current_thread_type(ThreadType.DM)

    try:
        # Create a block manually in DM thread context
        buf = [None, None]
        block = Block(
            buf, 2, Span(0, 2), (2, 1), BlockAcquisition.RESERVE, ThreadType.DM
        )

        # Create source tensor
        source_tensor = ttnn.Tensor(torch.ones((64, 32)))

        # Start copy - block should transition to NA state
        tx = copy(source_tensor, block)

        # Cannot read or write while copy is in progress (NAW state)
        # But also, block indexing is not allowed regardless of state
        with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
            _ = block[0]

        with pytest.raises(
            RuntimeError, match="Cannot write to Block.*copy lock error.*NAW"
        ):
            from python.sim import TILE_SHAPE

            # Need 2 items for block with span length 2
            block.store(
                [
                    ttnn.Tensor(torch.full(TILE_SHAPE, 5.0)),
                    ttnn.Tensor(torch.full(TILE_SHAPE, 6.0)),
                ]
            )

        # After tx.wait(), block becomes RW (can do more operations)
        tx.wait()

        # Block indexing is not allowed regardless of state
    finally:
        # Clean up thread context

        clear_current_thread_type()

    print("Copy sets block to NA state test passed!")


def test_push_validates_expected_state(api: DFBAPI) -> None:
    """Test that push() validates the block is in a valid state before completing.

    This test verifies that push() can only be called on reserve() blocks
    (not wait() blocks) and only when PUSH is in the expected operations.
    """

    set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_ones_tile()
        dfb = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        # Create a block in WAIT state (POP expected)
        # First, populate the DFB
        set_current_thread_type(ThreadType.DM)
        from python.sim.copy import copy

        src = make_ones_tile()
        blk = dfb.reserve()
        tx = copy(src, blk)
        tx.wait()
        blk.push()

        # Now wait for it in COMPUTE thread
        set_current_thread_type(ThreadType.COMPUTE)
        waited_block = dfb.wait()

        # Try to call push() on a wait() block - should fail
        # because waited_block is WAIT acquisition, not RESERVE
        # The error will first check expected operations (STORE_SRC vs PUSH)
        with pytest.raises(
            RuntimeError,
            match="Cannot perform push\\(\\): Expected one of \\[STORE_SRC\\], but got push\\(\\)",
        ):
            # Manually try to mark push (bypassing DFB's push method which checks pending_reserved_block)
            waited_block.mark_push_complete()

        # Clean up properly - use waited block as STORE_SRC before pop
        out_dfb = DataflowBuffer(
            element=element, shape=(1, 1), buffer_factor=2, api=api
        )
        out_block = out_dfb.reserve()
        out_block.store(waited_block)
        out_block.push()
        waited_block.pop()

        print("Push validates expected state test passed!")
    finally:

        clear_current_thread_type()


# ---------------------------------------------------------------------------
# DFBAPI low-level tests
# ---------------------------------------------------------------------------


def test_circular_buffer_basic_flow(configured_dfb8: Tuple[DFBAPI, DFBID]):
    api, dfb0 = configured_dfb8
    stats = api.dfb_stats(dfb0)
    assert stats.capacity == 8
    assert stats.visible == 0

    # Reserve and write 4 tiles
    api.dfb_reserve_back(dfb0, 4)
    ptr = api.get_write_ptr(dfb0)
    test_tensors = [make_full_tensor(32, 32, i + 1.0) for i in range(4)]
    ptr.store(test_tensors)
    api.dfb_push_back(dfb0, 4)
    stats = api.dfb_stats(dfb0)
    assert stats.visible == 4
    assert stats.free == 4

    # Wait and read
    api.dfb_wait_front(dfb0, 4)
    read_values = api.get_read_ptr(dfb0).to_list()
    assert len(read_values) == 4
    for i in range(4):
        val = read_values[i]
        assert val is not None
        assert tensors_exact_equal(val, test_tensors[i])
    api.dfb_pop_front(dfb0, 4)
    stats = api.dfb_stats(dfb0)
    assert stats.visible == 0

    # Reserve full capacity and write
    api.dfb_reserve_back(dfb0, 8)
    ptr = api.get_write_ptr(dfb0)
    test_tensors = [make_full_tensor(32, 32, float(i)) for i in range(8)]
    ptr.store(test_tensors)
    api.dfb_push_back(dfb0, 8)
    stats = api.dfb_stats(dfb0)
    assert stats.visible == 8

    # Cumulative wait and read
    api.dfb_wait_front(dfb0, 4)
    api.dfb_wait_front(dfb0, 8)
    read_values = api.get_read_ptr(dfb0).to_list()
    assert len(read_values) == 8
    for i in range(8):
        val = read_values[i]
        assert val is not None
        assert tensors_exact_equal(val, test_tensors[i])
    api.dfb_pop_front(dfb0, 8)
    stats = api.dfb_stats(dfb0)
    assert stats.visible == 0


def test_per_instance_timeout_effect():
    api = DFBAPI(timeout=0.2)
    dfb = 3
    api.host_configure_dfb(dfb, 4, shape=(1, 1))
    start = time.perf_counter()
    with pytest.raises(DFBTimeoutError, match="timed out after 0.2s"):
        api.dfb_wait_front(dfb, 1)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.4


def test_threaded_produce_consume(configured_dfb: Tuple[DFBAPI, DFBID]):
    api, dfb0 = configured_dfb
    result: List[List[DFBSlot]] = []

    def consumer():
        api.dfb_wait_front(dfb0, 4)
        result.append(api.get_read_ptr(dfb0).to_list())
        api.dfb_pop_front(dfb0, 4)

    t = threading.Thread(target=consumer)
    t.start()

    time.sleep(0.5)

    api.dfb_reserve_back(dfb0, 4)
    ptr = api.get_write_ptr(dfb0)
    test_tensors = [make_full_tensor(32, 32, 100.0 + i) for i in range(4)]
    ptr.store(test_tensors)
    api.dfb_push_back(dfb0, 4)
    t.join(timeout=1)

    assert len(result) == 1
    assert len(result[0]) == 4
    for i in range(4):
        val = result[0][i]
        assert val is not None
        assert tensors_exact_equal(val, test_tensors[i])


def test_dfb_pages_nonblocking(configured_dfb8: Tuple[DFBAPI, DFBID]):
    api, dfb2 = configured_dfb8

    assert not api.dfb_pages_available_at_front(dfb2, 1)
    assert api.dfb_pages_reservable_at_back(dfb2, 8)

    api.dfb_reserve_back(dfb2, 4)
    assert api.dfb_pages_reservable_at_back(dfb2, 4)

    ptr = api.get_write_ptr(dfb2)
    test_tensors = [make_full_tensor(32, 32, i + 1.0) for i in range(4)]
    ptr.store(test_tensors)
    api.dfb_push_back(dfb2, 4)
    assert api.dfb_pages_available_at_front(dfb2, 4)
    assert api.dfb_pages_available_at_front(dfb2, 2)

    api.dfb_wait_front(dfb2, 4)
    api.dfb_pop_front(dfb2, 4)
    assert not api.dfb_pages_available_at_front(dfb2, 1)


def test_dfb_pages_available_out_of_range_error(configured_dfb: Tuple[DFBAPI, DFBID]):
    api, dfb = configured_dfb
    with pytest.raises(DFBContractError, match="num_tiles must be <= capacity"):
        api.dfb_pages_available_at_front(dfb, 5)


def test_dfb_pages_reservable_out_of_range_error(configured_dfb: Tuple[DFBAPI, DFBID]):
    api, dfb = configured_dfb
    with pytest.raises(DFBContractError, match="num_tiles must be <= capacity"):
        api.dfb_pages_reservable_at_back(dfb, 5)


def test_dfb_pages_reservable_divisibility_error(configured_dfb8: Tuple[DFBAPI, DFBID]):
    api, dfb = configured_dfb8
    with pytest.raises(
        DFBContractError, match="First num_tiles=5 must evenly divide capacity=8"
    ):
        api.dfb_pages_reservable_at_back(dfb, 5)


def test_dfb_pages_available_divisibility_error(configured_dfb8: Tuple[DFBAPI, DFBID]):
    api, dfb = configured_dfb8
    api.dfb_reserve_back(dfb, 4)
    ptr = api.get_write_ptr(dfb)
    test_tensors = [make_full_tensor(32, 32, i + 1.0) for i in range(4)]
    ptr.store(test_tensors)
    api.dfb_push_back(dfb, 4)
    with pytest.raises(
        DFBContractError, match="First num_tiles=3 must evenly divide capacity=8"
    ):
        api.dfb_pages_available_at_front(dfb, 3)


def test_get_read_ptr_requires_wait(configured_dfb: Tuple[DFBAPI, DFBID]):
    api, dfb = configured_dfb
    with pytest.raises(
        DFBContractError, match="get_read_ptr requires prior dfb_wait_front"
    ):
        api.get_read_ptr(dfb)


def test_get_write_ptr_requires_reserve(configured_dfb: Tuple[DFBAPI, DFBID]):
    api, dfb = configured_dfb
    with pytest.raises(
        DFBContractError, match="get_write_ptr requires prior dfb_reserve_back"
    ):
        api.get_write_ptr(dfb)


def test_multiple_consumers_error(timeout_api: DFBAPI):
    api = timeout_api
    dfb = 0
    api.host_configure_dfb(dfb, 4, shape=(1, 1))
    errors: List[str] = []

    def consumer():
        try:
            api.dfb_wait_front(dfb, 4)
        except (DFBContractError, DFBTimeoutError) as e:
            errors.append(str(e))

    t1 = threading.Thread(target=consumer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert any(
        "Only one consumer thread may wait on a DFB at a time" in msg for msg in errors
    )


def test_multiple_producers_error(timeout_api: DFBAPI):
    api = timeout_api
    dfb = 0
    api.host_configure_dfb(dfb, 4, shape=(1, 1))
    errors: List[str] = []

    def producer():
        try:
            api.dfb_reserve_back(dfb, 4)
        except (DFBContractError, DFBTimeoutError) as e:
            errors.append(str(e))

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=producer)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert any(
        "Only one producer thread may reserve on a DFB at a time" in msg
        for msg in errors
    )


def test_allocate_dfb_id(api: DFBAPI):
    """Test that allocate_dfb_id allocates sequential IDs."""
    dfb_id0 = api.allocate_dfb_id()
    dfb_id1 = api.allocate_dfb_id()
    dfb_id2 = api.allocate_dfb_id()

    assert dfb_id0 == 0
    assert dfb_id1 == 1
    assert dfb_id2 == 2


def test_allocate_dfb_id_thread_safe(api: DFBAPI):
    """Test that allocate_dfb_id is thread-safe."""
    allocated_ids: List[DFBID] = []
    lock = threading.Lock()

    def allocate():
        dfb_id = api.allocate_dfb_id()
        with lock:
            allocated_ids.append(dfb_id)

    threads = [threading.Thread(target=allocate) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(allocated_ids) == 10
    assert len(set(allocated_ids)) == 10
    assert sorted(allocated_ids) == list(range(10))


def test_allocate_dfb_id_exceeds_max():
    """Test that allocating more than MAX_DFBS raises RuntimeError."""
    from python.sim.constants import MAX_DFBS

    api = DFBAPI()
    for _ in range(MAX_DFBS):
        api.allocate_dfb_id()

    with pytest.raises(
        RuntimeError, match=f"Maximum number of circular buffers exceeded: {MAX_DFBS}"
    ):
        api.allocate_dfb_id()


def test_heterogeneous_dfbs_in_same_api():
    """Test that a single DFBAPI instance can handle multiple circular buffers."""
    set_current_thread_type(ThreadType.COMPUTE)

    try:
        api = DFBAPI()
        element = make_full_tensor(32, 32, 1.0)

        dfb1 = DataflowBuffer(element=element, shape=(2, 2), buffer_factor=2, api=api)
        dfb2 = DataflowBuffer(element=element, shape=(2, 2), buffer_factor=2, api=api)

        write1 = dfb1.reserve()
        test_tensors_1 = [make_full_tensor(32, 32, i + 1.0) for i in range(len(write1))]
        write1.store(test_tensors_1)
        write1.push()

        read1 = dfb1.wait()
        write1_2 = dfb1.reserve()
        write1_2.store(read1)
        read1.pop()
        write1_2.push()

        write2 = dfb2.reserve()
        test_tensors_2 = [
            make_full_tensor(32, 32, i + 10.0) for i in range(len(write2))
        ]
        write2.store(test_tensors_2)
        write2.push()

        read2 = dfb2.wait()
        write2_2 = dfb2.reserve()
        write2_2.store(read2)
        read2.pop()
        write2_2.push()

        assert dfb1._api is api  # type: ignore
        assert dfb2._api is api  # type: ignore
    finally:
        set_current_thread_type(None)


def test_default_api_heterogeneous():
    """Test that an explicit API can handle multiple circular buffers."""
    set_current_thread_type(ThreadType.COMPUTE)

    try:
        api = DFBAPI()
        element = make_full_tensor(32, 32, 1.0)

        dfb1 = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)
        dfb2 = DataflowBuffer(element=element, shape=(1, 1), buffer_factor=2, api=api)

        assert dfb1._api is dfb2._api  # type: ignore
        assert dfb1._api is api  # type: ignore

        write1 = dfb1.reserve()
        write1.store([make_full_tensor(32, 32, 42.0)])
        write1.push()

        write2 = dfb2.reserve()
        write2.store([make_full_tensor(32, 32, 0.0)])
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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
