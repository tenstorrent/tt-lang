# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Block state machine enumerations and transition table.

Defines the thread-type context, access-state machine, and the full
transition table used by Block to validate correct usage patterns.
"""

from enum import Enum, auto
from typing import Dict, Optional, Tuple


# Global variable to track current thread type in cooperative scheduling
_current_thread_type: Optional["ThreadType"] = None


def get_current_thread_type() -> "ThreadType":
    """Get the current thread type.

    Returns:
        ThreadType

    Raises:
        RuntimeError: If thread type is not set (not within a thread context)
    """
    if _current_thread_type is None:
        raise RuntimeError(
            "Thread context not set. Must be called within a kernel thread or after "
            "calling set_current_thread_type()."
        )
    return _current_thread_type


def set_current_thread_type(thread_type: Optional["ThreadType"]) -> None:
    """Set the current thread type.

    Args:
        thread_type: The thread type to set, or None to clear the context
    """
    global _current_thread_type
    _current_thread_type = thread_type


def clear_current_thread_type() -> None:
    """Clear the current thread type."""
    global _current_thread_type
    _current_thread_type = None


class AccessState(Enum):
    """Access state for a block in the state machine."""

    MW = (
        auto()
    )  # Must be Written: block was reserved and contains garbage data, must be written to
    MR = (
        auto()
    )  # Must be Read: block was waited on or written to and never read, must be read from or pushed
    RW = (
        auto()
    )  # Read-Write: block was waited on or written to (MR) and then read from, can be read more or overwritten
    A = (
        auto()
    )  # Accumulate: block has been accumulated to, can continue accumulating or must be read or pushed
    NAR = auto()  # No Access while Reading: block is being asynchronously read from
    NAW = auto()  # No Access while Writing: block is being asynchronously written to
    OS = auto()  # Out of Scope: block was pushed or popped


class ThreadType(Enum):
    """Thread type for block operations."""

    DM = auto()  # Data Movement
    COMPUTE = auto()  # Compute


class BlockAcquisition(Enum):
    """How the block was acquired."""

    RESERVE = auto()  # Via reserve()
    WAIT = auto()  # Via wait()


class ExpectedOp(Enum):
    """Expected next operation on a block."""

    COPY_SRC = auto()  # Expect copy(blk, ...) - block as source
    COPY_DST = auto()  # Expect copy(..., blk) - block as destination
    TX_WAIT = auto()  # Expect tx.wait()
    PUSH = auto()  # Expect blk.push()
    POP = auto()  # Expect blk.pop()
    STORE = (
        auto()
    )  # Expect blk.store(...) - block as destination, regular store (acc=False)
    STORE_ACC = (
        auto()
    )  # Expect blk.store(..., acc=True) - block as destination, accumulator store
    STORE_SRC = (
        auto()
    )  # Expect other_blk.store(blk, ...) - block as source/input to store
    DONE = auto()  # No more operations expected


# State machine transition table
# Organized by (acquisition, thread_type) -> {(operation, access_state): (new_access_state, new_expected_ops)}
# This structure makes it easy to see all transitions for a particular acquisition/thread combination
STATE_TRANSITIONS: Dict[
    Tuple[BlockAcquisition, ThreadType],
    Dict[
        Tuple[str, AccessState],
        Tuple[AccessState, set[ExpectedOp]],
    ],
] = {
    # DM thread, WAIT acquisition
    (BlockAcquisition.WAIT, ThreadType.DM): {
        # Copy as source: MR/RW -> NAR + TX_WAIT
        ("copy_src", AccessState.MR): (
            AccessState.NAR,
            {ExpectedOp.TX_WAIT},
        ),
        ("copy_src", AccessState.RW): (
            AccessState.NAR,
            {ExpectedOp.TX_WAIT},
        ),
        # Copy as destination: RW -> NAW + TX_WAIT
        ("copy_dst", AccessState.RW): (
            AccessState.NAW,
            {ExpectedOp.TX_WAIT},
        ),
        # TX wait complete from NAR -> RW with copy + pop ops
        ("tx_wait", AccessState.NAR): (
            AccessState.RW,
            {ExpectedOp.COPY_DST, ExpectedOp.COPY_SRC, ExpectedOp.POP},
        ),
        # TX wait complete from NAW -> MR with copy_src only
        ("tx_wait", AccessState.NAW): (
            AccessState.MR,
            {ExpectedOp.COPY_SRC},
        ),
    },
    # DM thread, RESERVE acquisition
    (BlockAcquisition.RESERVE, ThreadType.DM): {
        # Copy as source: MR/RW -> NAR + TX_WAIT
        ("copy_src", AccessState.MR): (
            AccessState.NAR,
            {ExpectedOp.TX_WAIT},
        ),
        ("copy_src", AccessState.RW): (
            AccessState.NAR,
            {ExpectedOp.TX_WAIT},
        ),
        # Copy as destination: MW/RW -> NAW + TX_WAIT
        ("copy_dst", AccessState.MW): (
            AccessState.NAW,
            {ExpectedOp.TX_WAIT},
        ),
        ("copy_dst", AccessState.RW): (
            AccessState.NAW,
            {ExpectedOp.TX_WAIT},
        ),
        # TX wait complete from NAW -> MR with push + copy_src
        ("tx_wait", AccessState.NAW): (
            AccessState.MR,
            {ExpectedOp.PUSH, ExpectedOp.COPY_SRC},
        ),
        # TX wait complete from NAR -> RW with all copy ops + push
        ("tx_wait", AccessState.NAR): (
            AccessState.RW,
            {ExpectedOp.COPY_DST, ExpectedOp.COPY_SRC, ExpectedOp.PUSH},
        ),
    },
    # COMPUTE thread, WAIT acquisition
    (BlockAcquisition.WAIT, ThreadType.COMPUTE): {
        # Store read complete: MR/RW/A -> RW with store ops + pop
        ("store_src", AccessState.MR): (
            AccessState.RW,
            {
                ExpectedOp.STORE_SRC,
                ExpectedOp.STORE,
                ExpectedOp.STORE_ACC,
                ExpectedOp.POP,
            },
        ),
        ("store_src", AccessState.RW): (
            AccessState.RW,
            {
                ExpectedOp.STORE_SRC,
                ExpectedOp.STORE,
                ExpectedOp.STORE_ACC,
                ExpectedOp.POP,
            },
        ),
        ("store_src", AccessState.A): (
            AccessState.RW,
            {
                ExpectedOp.STORE_SRC,
                ExpectedOp.STORE,
                ExpectedOp.STORE_ACC,
                ExpectedOp.POP,
            },
        ),
        # Store accumulate complete: RW/A -> A with store_src + store_acc
        ("store_dst_acc", AccessState.RW): (
            AccessState.A,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE_ACC},
        ),
        ("store_dst_acc", AccessState.A): (
            AccessState.A,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE_ACC},
        ),
        # Store regular complete: RW -> MR with store_src only
        ("store_dst", AccessState.RW): (
            AccessState.MR,
            {ExpectedOp.STORE_SRC},
        ),
    },
    # COMPUTE thread, RESERVE acquisition
    (BlockAcquisition.RESERVE, ThreadType.COMPUTE): {
        # Store read complete: MR/RW/A -> RW with store ops + push
        ("store_src", AccessState.MR): (
            AccessState.RW,
            {
                ExpectedOp.STORE_SRC,
                ExpectedOp.STORE,
                ExpectedOp.STORE_ACC,
                ExpectedOp.PUSH,
            },
        ),
        ("store_src", AccessState.RW): (
            AccessState.RW,
            {
                ExpectedOp.STORE_SRC,
                ExpectedOp.STORE,
                ExpectedOp.STORE_ACC,
                ExpectedOp.PUSH,
            },
        ),
        ("store_src", AccessState.A): (
            AccessState.RW,
            {
                ExpectedOp.STORE_SRC,
                ExpectedOp.STORE,
                ExpectedOp.STORE_ACC,
                ExpectedOp.PUSH,
            },
        ),
        # Store accumulate complete: MW/RW/A -> A with store_src + store_acc + push
        ("store_dst_acc", AccessState.MW): (
            AccessState.A,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE_ACC, ExpectedOp.PUSH},
        ),
        ("store_dst_acc", AccessState.RW): (
            AccessState.A,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE_ACC, ExpectedOp.PUSH},
        ),
        ("store_dst_acc", AccessState.A): (
            AccessState.A,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE_ACC, ExpectedOp.PUSH},
        ),
        # Store regular complete: MW/RW -> MR with store_src + push
        ("store_dst", AccessState.MW): (
            AccessState.MR,
            {ExpectedOp.STORE_SRC, ExpectedOp.PUSH},
        ),
        ("store_dst", AccessState.RW): (
            AccessState.MR,
            {ExpectedOp.STORE_SRC, ExpectedOp.PUSH},
        ),
    },
}
