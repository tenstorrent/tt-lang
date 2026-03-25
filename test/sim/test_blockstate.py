# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for BlockStateMachine and Block access-state enforcement.

Covers state transitions, access restrictions, ROR(N) multi-copy counting,
push/pop validation, and NAW copy-destination locking.
"""

import pytest
import torch
from test_utils import (
    make_element_for_buffer_shape,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
)

from python.sim import TILE_SHAPE, copy, ttnn
from python.sim.blockstate import (
    BlockAcquisition,
    ThreadType,
)
from python.sim.context import (
    clear_current_thread_type,
    set_current_thread_type,
)
from python.sim.dfb import Block, DataflowBuffer


@pytest.fixture(autouse=True)
def setup_thread_context(compute_thread_context):
    """Set up scheduler and COMPUTE thread context for all blockstate tests."""
    pass


# ---------------------------------------------------------------------------
# Basic state machine restrictions
# ---------------------------------------------------------------------------


def test_block_state_machine_restrictions() -> None:
    """Test that block state machine enforces access restrictions."""
    element = make_zeros_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

    # Test: Cannot index blocks - block indexing is not allowed
    block = dfb.reserve()

    # Attempting to index block should fail
    with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
        _ = block[0]

    # Store makes it RO (for regular store) or RW (for acc store)
    block.store(Block.from_tensor(ttnn.Tensor(torch.full(TILE_SHAPE, 5.0))), acc=True)

    block.push()

    # Test: Cannot write to RO (Read-Only) state after wait()
    read_block = dfb.wait()

    # Cannot write - wait() blocks expect STORE_SRC, not STORE
    with pytest.raises(RuntimeError, match="Cannot perform store.*Expected one of"):
        read_block.store(Block.from_tensor(ttnn.Tensor(torch.full(TILE_SHAPE, 10.0))))

    # Use waited block as STORE_SRC before pop
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
    out_block = out_dfb.reserve()
    out_block.store(read_block)
    out_block.push()
    read_block.pop()


def test_copy_sets_block_to_na_state() -> None:
    """Test that copy operations set blocks to NAW (No Access while Writing) state."""
    set_current_thread_type(ThreadType.DM)

    try:
        block = Block(
            ttnn.Tensor(torch.zeros((64, 32))),
            (2, 1),
            BlockAcquisition.RESERVE,
            ThreadType.DM,
        )

        source_tensor = ttnn.Tensor(torch.ones((64, 32)))

        tx = copy(source_tensor, block)

        # Block indexing is never allowed
        with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
            _ = block[0]

        # Block is locked as copy destination (NAW state)
        with pytest.raises(
            RuntimeError, match="Cannot write to Block.*copy lock error.*NAW"
        ):
            block.store(Block.from_tensor(ttnn.Tensor(torch.ones((64, 32)))))

        tx.wait()
    finally:
        clear_current_thread_type()


def test_push_validates_expected_state() -> None:
    """Test that push() validates the block is in a valid state before completing."""
    set_current_thread_type(ThreadType.COMPUTE)

    try:
        element = make_ones_tile()
        dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)

        # Populate the DFB from a DM thread
        set_current_thread_type(ThreadType.DM)
        from python.sim.copy import copy as dm_copy

        src = make_ones_tile()
        blk = dfb.reserve()
        tx = dm_copy(src, blk)
        tx.wait()
        blk.push()

        # Now wait for it in COMPUTE thread
        set_current_thread_type(ThreadType.COMPUTE)
        waited_block = dfb.wait()

        # push() on a wait() block must fail: STORE_SRC is expected, not PUSH
        with pytest.raises(
            RuntimeError,
            match="Cannot perform push\\(\\): Expected one of \\[STORE_SRC\\], but got push\\(\\)",
        ):
            waited_block.mark_push_complete()

        # Clean up properly
        out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), buffer_factor=2)
        out_block = out_dfb.reserve()
        out_block.store(waited_block)
        out_block.push()
        waited_block.pop()
    finally:
        clear_current_thread_type()


# ---------------------------------------------------------------------------
# ROR(N) state: multiple concurrent copies from a single source block
# ---------------------------------------------------------------------------


class TestRORState:
    """Test ROR(N) state: multiple concurrent copies from a single source block.

    When a block is used as a copy source it enters ROR(N=1). Each additional
    copy as source increments N; each tx.wait() decrements N. The block
    transitions to RW only when the last outstanding copy completes (N==1 -> RW).
    """

    def _make_wait_block(self) -> Block:
        """Return a DM WAIT block pre-loaded with data via a DFB reserve/push cycle."""
        set_current_thread_type(ThreadType.DM)
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            buffer_factor=2,
        )
        with dfb.reserve() as blk:
            tx = copy(make_rand_tensor(64, 32), blk)
            tx.wait()
        return dfb.wait().__enter__()

    def test_ror_entered_on_first_copy(self) -> None:
        """First copy_src transitions the block from MR to ROR(N=1)."""
        block = self._make_wait_block()
        assert block._access_state.name == "MR"

        tx = copy(block, make_rand_tensor(64, 32))
        assert block._access_state.name == "ROR"
        assert block._sm.ror_count == 1

        tx.wait()

    def test_ror_count_increments_on_each_additional_copy(self) -> None:
        """Each new copy launched from ROR increments N without leaving ROR."""
        block = self._make_wait_block()

        tx1 = copy(block, make_rand_tensor(64, 32))
        assert block._sm.ror_count == 1

        tx2 = copy(block, make_rand_tensor(64, 32))
        assert block._access_state.name == "ROR"
        assert block._sm.ror_count == 2

        tx3 = copy(block, make_rand_tensor(64, 32))
        assert block._access_state.name == "ROR"
        assert block._sm.ror_count == 3

        tx1.wait()
        tx2.wait()
        tx3.wait()

    def test_write_blocked_throughout_all_copies_in_flight(self) -> None:
        """Writing to a source block is forbidden for every copy still outstanding."""
        block = self._make_wait_block()

        tx1 = copy(block, make_rand_tensor(64, 32))
        tx2 = copy(block, make_rand_tensor(64, 32))

        for _ in range(2):
            with pytest.raises(RuntimeError, match="has no access.*ROR state"):
                block.store(Block.from_tensor(make_rand_tensor(64, 32)))

        tx1.wait()
        with pytest.raises(RuntimeError, match="has no access.*ROR state"):
            block.store(Block.from_tensor(make_rand_tensor(64, 32)))

        tx2.wait()

    def test_ror_count_decrements_on_each_wait_stays_in_ror(self) -> None:
        """Each tx.wait() decrements N; block stays in ROR until N reaches 1."""
        block = self._make_wait_block()

        tx1 = copy(block, make_rand_tensor(64, 32))
        tx2 = copy(block, make_rand_tensor(64, 32))
        tx3 = copy(block, make_rand_tensor(64, 32))
        assert block._sm.ror_count == 3

        tx1.wait()
        assert block._access_state.name == "ROR"
        assert block._sm.ror_count == 2

        tx2.wait()
        assert block._access_state.name == "ROR"
        assert block._sm.ror_count == 1

        # Last wait must transition to RW
        tx3.wait()
        assert block._access_state.name == "RW"

    def test_last_wait_transitions_to_rw(self) -> None:
        """When the final outstanding copy completes, the block enters RW."""
        block = self._make_wait_block()

        tx1 = copy(block, make_rand_tensor(64, 32))
        tx2 = copy(block, make_rand_tensor(64, 32))

        tx1.wait()
        assert block._access_state.name == "ROR"

        tx2.wait()
        assert block._access_state.name == "RW"

    def test_can_launch_another_copy_after_partial_waits(self) -> None:
        """New copies can be launched from ROR while other copies are still in flight."""
        block = self._make_wait_block()

        tx1 = copy(block, make_rand_tensor(64, 32))
        tx2 = copy(block, make_rand_tensor(64, 32))
        assert block._sm.ror_count == 2

        tx1.wait()
        assert block._sm.ror_count == 1

        # Launch another copy while the first has completed but tx2 is still pending
        tx3 = copy(block, make_rand_tensor(64, 32))
        assert block._access_state.name == "ROR"
        assert block._sm.ror_count == 2

        tx2.wait()
        assert block._sm.ror_count == 1

        tx3.wait()
        assert block._access_state.name == "RW"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
