# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tests for BlockStateMachine and Block access-state enforcement.

Covers state transitions, access restrictions, ROR(N) multi-copy counting,
push/pop validation, NAW copy-destination locking, and user-facing error
message content from the block state helpers and DataflowBuffer.
"""

import pytest
import torch
from test_utils import (
    make_element_for_buffer_shape,
    make_ones_tile,
    make_rand_tensor,
    make_zeros_tile,
)

from sim import TILE_SHAPE, copy, ttnn
from sim.blockstate import (
    AccessState,
    BlockAcquisition,
    ExpectedOp,
    KernelType,
    format_block_finished_error,
    format_cannot_read_block,
    format_cannot_write_block,
    format_validate_mismatch,
)
from sim.context import (
    clear_current_kernel_type,
    set_current_kernel_type,
)
from sim.copy import copy as dm_copy
from sim.dfb import Block, DataflowBuffer


@pytest.fixture(autouse=True)
def setup_kernel_context(compute_kernel_context):
    """Set up scheduler and COMPUTE kernel context for all blockstate tests."""
    pass


# ---------------------------------------------------------------------------
# Basic state machine restrictions
# ---------------------------------------------------------------------------


def test_block_state_machine_restrictions() -> None:
    """Test that block state machine enforces access restrictions."""
    element = make_zeros_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)

    # Test: Cannot index blocks - block indexing is not allowed
    block = dfb.reserve()

    # Attempting to index block should fail
    with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
        _ = block[0]

    # Store makes it MR (read-only after regular store)
    block.store(Block.from_tensor(ttnn.Tensor(torch.full(TILE_SHAPE, 5.0))))

    block.push()

    # Test: Cannot write to RO (Read-Only) state after wait()
    read_block = dfb.wait()

    # Cannot write - wait() blocks expect STORE_SRC, not STORE
    with pytest.raises(
        RuntimeError,
        match=r"(?s)Cannot perform store\(\): not a valid next dataflow step.*\[STORE_SRC\]",
    ):
        read_block.store(Block.from_tensor(ttnn.Tensor(torch.full(TILE_SHAPE, 10.0))))

    # Use waited block as STORE_SRC before pop
    out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)
    out_block = out_dfb.reserve()
    out_block.store(read_block)
    out_block.push()
    read_block.pop()


def test_copy_sets_block_to_na_state() -> None:
    """Test that copy operations set blocks to NAW (No Access while Writing) state."""
    set_current_kernel_type(KernelType.DM)

    try:
        block = Block(
            ttnn.Tensor(torch.zeros((64, 32))),
            (2, 1),
            BlockAcquisition.RESERVE,
            KernelType.DM,
        )

        source_tensor = ttnn.Tensor(torch.ones((64, 32)))

        tx = copy(source_tensor, block)

        # Block indexing is never allowed
        with pytest.raises(RuntimeError, match="Block indexing.*not allowed"):
            _ = block[0]

        # Block is locked as copy destination (NAW state)
        with pytest.raises(
            RuntimeError,
            match=r"(?s)Cannot write to this buffer block.*NAW.*copy lock error",
        ):
            block.store(Block.from_tensor(ttnn.Tensor(torch.ones((64, 32)))))

        tx.wait()
    finally:
        clear_current_kernel_type()


def test_push_validates_expected_state() -> None:
    """Test that push() validates the block is in a valid state before completing."""
    set_current_kernel_type(KernelType.COMPUTE)

    try:
        element = make_ones_tile()
        dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)

        # Populate the DFB from a DM kernel
        set_current_kernel_type(KernelType.DM)
        from sim.copy import copy as dm_copy

        src = make_ones_tile()
        blk = dfb.reserve()
        tx = dm_copy(src, blk)
        tx.wait()
        blk.push()

        # Now wait for it in COMPUTE kernel
        set_current_kernel_type(KernelType.COMPUTE)
        waited_block = dfb.wait()

        # push() on a wait() block must fail: STORE_SRC is expected, not PUSH
        with pytest.raises(
            RuntimeError,
            match=r"(?s)Cannot perform push\(\): not a valid next dataflow step.*\[STORE_SRC\].*attempted PUSH",
        ):
            waited_block.mark_push_complete()

        # Clean up properly
        out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)
        out_block = out_dfb.reserve()
        out_block.store(waited_block)
        out_block.push()
        waited_block.pop()
    finally:
        clear_current_kernel_type()


# ---------------------------------------------------------------------------
# assign_src: arithmetic use of wait() block before store() fires
# ---------------------------------------------------------------------------


class TestAssignSrcTransition:
    """Tests for the assign_src state machine transition.

    When a WAIT/COMPUTE block is used as an arithmetic operand (assigned to a
    temporary), assign_src fires: the block moves from MR to RW and POP is
    unlocked.  A pending store confirmation is registered on the DFB and must
    be cleared by mark_store_read_complete() before program termination.
    """

    def _make_compute_wait_block(
        self,
    ) -> tuple["DataflowBuffer", "Block"]:
        """Return a DFB and a WAIT/COMPUTE block ready for use."""
        set_current_kernel_type(KernelType.DM)
        element = make_ones_tile()
        dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)
        from sim.copy import copy as dm_copy

        src = make_ones_tile()
        blk = dfb.reserve()
        tx = dm_copy(src, blk)
        tx.wait()
        blk.push()

        set_current_kernel_type(KernelType.COMPUTE)
        waited = dfb.wait(name="compute_in")
        return dfb, waited

    def test_assign_src_unlocks_pop(self) -> None:
        """assign_src moves the block from MR to RW and allows pop()."""
        dfb, block = self._make_compute_wait_block()
        try:
            assert block.access_state.name == "MR"
            block.mark_assign_src_complete()
            assert block.access_state.name == "RW"
            assert ExpectedOp.POP in block.expected_ops
        finally:
            block.pop()
            clear_current_kernel_type()

    def test_assign_src_registers_pending_confirmation(self) -> None:
        """assign_src adds the block to the DFB's pending confirmation set."""
        dfb, block = self._make_compute_wait_block()
        try:
            assert block not in dfb._pending_confirmations
            block.mark_assign_src_complete()
            assert block in dfb._pending_confirmations
        finally:
            block.pop()
            clear_current_kernel_type()

    def test_store_read_complete_clears_pending_confirmation(self) -> None:
        """mark_store_read_complete() clears the pending confirmation registered by assign_src."""
        dfb, block = self._make_compute_wait_block()
        try:
            block.mark_assign_src_complete()
            assert block in dfb._pending_confirmations
            block.mark_store_read_complete()
            assert block not in dfb._pending_confirmations
        finally:
            block.pop()
            clear_current_kernel_type()

    def test_assign_src_idempotent_on_rw(self) -> None:
        """assign_src is idempotent: calling it again from RW state is safe."""
        dfb, block = self._make_compute_wait_block()
        try:
            block.mark_assign_src_complete()
            assert block.access_state.name == "RW"
            block.mark_assign_src_complete()
            assert block.access_state.name == "RW"
            assert len(dfb._pending_confirmations) == 1
        finally:
            block.pop()
            clear_current_kernel_type()

    def test_validate_no_pending_blocks_raises_on_unconfirmed(self) -> None:
        """validate_no_pending_blocks() raises if a block was assigned but never stored."""
        dfb, block = self._make_compute_wait_block()
        try:
            block.mark_assign_src_complete()
            block.pop()
            with pytest.raises(RuntimeError) as err:
                dfb.validate_no_pending_blocks()
            msg = str(err.value)
            assert "never reached a store" in msg
            assert "block_name='compute_in'" in msg
        finally:
            clear_current_kernel_type()


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
        set_current_kernel_type(KernelType.DM)
        dfb = DataflowBuffer(
            likeness_tensor=make_element_for_buffer_shape((2, 1)),
            shape=(2, 1),
            block_count=2,
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
            with pytest.raises(
                RuntimeError, match="Cannot write to this buffer block.*ROR"
            ):
                block.store(Block.from_tensor(make_rand_tensor(64, 32)))

        tx1.wait()
        with pytest.raises(
            RuntimeError, match="Cannot write to this buffer block.*ROR"
        ):
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


# ---------------------------------------------------------------------------
# User-facing error message text (format helpers + DFB)
# ---------------------------------------------------------------------------


def test_format_validate_mismatch_includes_what_to_do_and_details() -> None:
    """The mismatch template always carries a lead-in, follow-up, and a Details line."""
    msg = format_validate_mismatch(
        "store()",
        ExpectedOp.STORE,
        {ExpectedOp.STORE_SRC, ExpectedOp.POP},
        AccessState.MR,
        BlockAcquisition.WAIT,
        KernelType.COMPUTE,
    )
    assert "Next:" in msg
    assert "Details: expected one of" in msg
    assert "attempted STORE" in msg
    assert "acquisition=WAIT" in msg
    assert "kernel=COMPUTE" in msg
    assert "access=MR" in msg


def test_format_validate_mismatch_store_on_waited_block_mentions_out_block_store() -> (
    None
):
    """A common failure mode is store() on a wait() block: hint names the right pattern."""
    msg = format_validate_mismatch(
        "store()",
        ExpectedOp.STORE,
        {ExpectedOp.STORE_SRC},
        AccessState.MR,
        BlockAcquisition.WAIT,
        KernelType.COMPUTE,
    )
    assert "wait() block" in msg
    assert "out_block.store" in msg


def test_format_validate_mismatch_push_on_wait_block_mentions_pop_not_push() -> None:
    """push() on a wait() block: hint distinguishes reserve vs wait shutdown."""
    msg = format_validate_mismatch(
        "push()",
        ExpectedOp.PUSH,
        {ExpectedOp.STORE_SRC},
        AccessState.MR,
        BlockAcquisition.WAIT,
        KernelType.COMPUTE,
    )
    assert "push() is for reserve() only" in msg
    assert "pop()" in msg


def test_format_validate_mismatch_naw_mentions_in_flight_and_wait() -> None:
    """NAW errors point at the async copy and the need to wait."""
    msg = format_validate_mismatch(
        "store()",
        ExpectedOp.STORE,
        {ExpectedOp.TX_WAIT},
        AccessState.NAW,
        BlockAcquisition.RESERVE,
        KernelType.DM,
    )
    assert (
        "may still be in flight" in msg.lower() or "wait until the copy" in msg.lower()
    )


def test_format_validate_mismatch_includes_copy_callsite_when_pending_provided() -> (
    None
):
    """Mismatch errors during NAW/ROR include the recorded copy(...) callsite when available."""
    msg = format_validate_mismatch(
        "store()",
        ExpectedOp.STORE,
        {ExpectedOp.TX_WAIT},
        AccessState.NAW,
        BlockAcquisition.RESERVE,
        KernelType.DM,
        pending_copy_location=("/x/y/z.py", 12),
    )
    assert "Where: the copy involving this block was requested at /x/y/z.py:12" in msg


def test_format_block_finished_error_mentions_reacquire_and_push_pop() -> None:
    """Finished / inactive blocks tell the user to get a new block, not to reuse the handle."""
    msg = format_block_finished_error("store()", AccessState.OS)
    assert "no longer active" in msg
    assert "reserve()" in msg
    assert "wait()" in msg
    assert "expected-ops=empty" in msg


def test_format_cannot_read_mw_and_naw_explain_cause() -> None:
    """Read guards name MW vs NAW and suggest fill/wait as appropriate."""
    msg_mw = format_cannot_read_block(
        AccessState.MW, {ExpectedOp.COPY_DST}, BlockAcquisition.RESERVE
    )
    assert "this buffer block" in msg_mw
    assert "MW" in msg_mw
    assert "Next:" in msg_mw
    assert "state=MW" in msg_mw

    msg_naw = format_cannot_read_block(
        AccessState.NAW, {ExpectedOp.TX_WAIT}, BlockAcquisition.RESERVE
    )
    assert "this buffer block" in msg_naw
    assert "NAW" in msg_naw
    assert "in flight" in msg_naw.lower()
    assert "wait for that copy" in msg_naw.lower()


def test_format_cannot_write_naw_includes_copy_lock_phrase() -> None:
    """NAW write errors stay aligned with copy_lock_error example and tests."""
    msg = format_cannot_write_block(AccessState.NAW, {ExpectedOp.TX_WAIT})
    assert "this buffer block" in msg
    assert "copy lock error" in msg
    assert "state=NAW" in msg
    assert "copy" in msg.lower() and "wait" in msg.lower()


def test_format_cannot_write_naw_includes_pending_copy_callsite_when_provided() -> None:
    """NAW messages name the user callsite of copy(..., block) when available."""
    msg = format_cannot_write_block(
        AccessState.NAW,
        {ExpectedOp.TX_WAIT},
        pending_copy_location=("/src/kernel.py", 99),
    )
    assert "Where: copy into this block was requested at /src/kernel.py:99" in msg


def test_format_cannot_write_ror_mentions_further_tx_waits() -> None:
    """ROR write rejection explains outstanding copies, not a bare enum dump."""
    msg = format_cannot_write_block(
        AccessState.ROR, {ExpectedOp.COPY_SRC, ExpectedOp.TX_WAIT}
    )
    assert "this buffer block" in msg
    assert "ROR" in msg
    assert "in-flight" in msg.lower()
    assert "wait" in msg.lower()


def test_format_cannot_write_ror_includes_pending_copy_callsite_when_provided() -> None:
    """ROR messages name the user callsite of copy(block, ...) when available."""
    msg = format_cannot_write_block(
        AccessState.ROR,
        {ExpectedOp.COPY_SRC, ExpectedOp.TX_WAIT},
        pending_copy_location=("/src/k.py", 5),
    )
    assert "Where: copy from this block was requested at /src/k.py:5" in msg


def test_bsm_validate_finished_block_uses_block_finished_error() -> None:
    """Empty expected-ops should surface the reacquire / lifecycle message."""
    dfb = DataflowBuffer(likeness_tensor=make_zeros_tile(), shape=(1, 1), block_count=2)
    b = dfb.reserve()
    b.store(Block.from_tensor(ttnn.Tensor(torch.full(TILE_SHAPE, 1.0))))
    b.push()
    with pytest.raises(RuntimeError) as err:
        b.mark_store_complete()
    msg = str(err.value)
    assert "no longer active" in msg
    assert "Next:" in msg


def test_block_cannot_read_mw_uses_friendly_read_message(
    dm_kernel_context,
) -> None:  # noqa: ARG001
    """Reading before the first write uses format_cannot_read (MW) wording."""
    dfb = DataflowBuffer(likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2)
    b = dfb.reserve(name="read_test")
    with pytest.raises(RuntimeError) as err:
        _ = b.get_item(0)
    m = str(err.value)
    assert "Cannot read from this buffer block" in m
    assert "name: 'read_test'" in m
    assert "MW" in m


def test_validate_no_pending_reserved_mentions_push_and_incomplete(
    dm_kernel_context,
) -> None:  # noqa: ARG001
    """A held DM reserve() without release should surface a simulator-bug blurb."""
    dfb = DataflowBuffer(likeness_tensor=make_ones_tile(), shape=(1, 1), block_count=2)
    dfb.reserve(name="held_buf")
    with pytest.raises(RuntimeError) as err:
        dfb.validate_no_pending_blocks()
    msg = str(err.value)
    assert "incomplete or unconsumed" in msg
    assert "simulator bug" in msg
    assert "reserve()" in msg
    assert "auto-push" in msg
    assert "1)" in msg
    assert "held_buf" in msg
    assert "block_name=" in msg
    assert "name: 'held_buf'" in msg


def test_validate_no_pending_wait_mentions_pop_and_incomplete(
    dm_kernel_context,
) -> None:  # noqa: ARG001
    """A held wait() without pop() should emit a simulator-bug blurb."""
    element = make_ones_tile()
    dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)
    blk = dfb.reserve()
    tx = dm_copy(ttnn.Tensor(torch.full(TILE_SHAPE, 1.0)), blk)
    tx.wait()
    blk.push()

    set_current_kernel_type(KernelType.COMPUTE)
    try:
        waited = dfb.wait(name="consumer_view")
        with pytest.raises(RuntimeError) as err:
            dfb.validate_no_pending_blocks()
        msg = str(err.value)
        assert "incomplete or unconsumed" in msg
        assert "simulator bug" in msg
        assert "wait()" in msg
        assert "auto-pop" in msg
        assert "1)" in msg
        assert "consumer_view" in msg
        assert "block_name=" in msg
        assert "name: 'consumer_view'" in msg
    finally:
        # WAIT compute blocks require store-as-source before pop(); drain then release the slot.
        out_dfb = DataflowBuffer(likeness_tensor=element, shape=(1, 1), block_count=2)
        out_block = out_dfb.reserve()
        out_block.store(waited)
        out_block.push()
        waited.pop()
        clear_current_kernel_type()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
