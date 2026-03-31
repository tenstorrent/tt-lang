# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Block state machine enumerations and transition table.

Defines the thread-type context, access-state machine, and the full
transition table used by Block to validate correct usage patterns.
"""

from enum import Enum, auto
from typing import Dict, Set, Tuple


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
    ROR = (
        auto()
    )  # Read-Only while Reading: block has N copies in flight; N is tracked separately
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
    STORE = auto()  # Expect blk.store(...) - block as destination
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
        # Copy as source: MR/RW -> ROR; further copies and tx_wait both expected
        ("copy_src", AccessState.MR): (
            AccessState.ROR,
            {ExpectedOp.TX_WAIT, ExpectedOp.COPY_SRC},
        ),
        ("copy_src", AccessState.RW): (
            AccessState.ROR,
            {ExpectedOp.TX_WAIT, ExpectedOp.COPY_SRC},
        ),
        # Copy as destination: RW -> NAW + TX_WAIT
        ("copy_dst", AccessState.RW): (
            AccessState.NAW,
            {ExpectedOp.TX_WAIT},
        ),
        # TX wait complete from ROR (N==1) -> RW with copy + pop ops
        ("tx_wait", AccessState.ROR): (
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
        # Copy as source: MR/RW -> ROR; further copies and tx_wait both expected
        ("copy_src", AccessState.MR): (
            AccessState.ROR,
            {ExpectedOp.TX_WAIT, ExpectedOp.COPY_SRC},
        ),
        ("copy_src", AccessState.RW): (
            AccessState.ROR,
            {ExpectedOp.TX_WAIT, ExpectedOp.COPY_SRC},
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
        # TX wait complete from ROR (N==1) -> RW with all copy ops + push
        ("tx_wait", AccessState.ROR): (
            AccessState.RW,
            {ExpectedOp.COPY_DST, ExpectedOp.COPY_SRC, ExpectedOp.PUSH},
        ),
    },
    # COMPUTE thread, WAIT acquisition
    (BlockAcquisition.WAIT, ThreadType.COMPUTE): {
        # Assign as arithmetic source: MR/RW -> RW; POP now allowed but store
        # confirmation is deferred and tracked until program termination.
        ("assign_src", AccessState.MR): (
            AccessState.RW,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.POP},
        ),
        ("assign_src", AccessState.RW): (
            AccessState.RW,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.POP},
        ),
        # Store read complete: MR/RW -> RW with store ops + pop
        ("store_src", AccessState.MR): (
            AccessState.RW,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.POP},
        ),
        ("store_src", AccessState.RW): (
            AccessState.RW,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.POP},
        ),
        # Store complete: RW -> MR with store_src only
        ("store_dst", AccessState.RW): (
            AccessState.MR,
            {ExpectedOp.STORE_SRC},
        ),
    },
    # COMPUTE thread, RESERVE acquisition
    (BlockAcquisition.RESERVE, ThreadType.COMPUTE): {
        # Store read complete: MR/RW -> RW with store ops + push
        ("store_src", AccessState.MR): (
            AccessState.RW,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.PUSH},
        ),
        ("store_src", AccessState.RW): (
            AccessState.RW,
            {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.PUSH},
        ),
        # Store complete: MW/RW -> MR with store_src + push
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

# ROR expected-ops set, shared by all in-state ROR transitions.
_ROR_EXPECTED: Set[ExpectedOp] = {ExpectedOp.TX_WAIT, ExpectedOp.COPY_SRC}


class BlockStateMachine:
    """All access-state logic for a Block: initial state, validation, and transitions.

    Owns the five state fields (acquisition, thread_type, access_state, expected_ops,
    ror_count) and every method that mutates them.  Block in dfb.py holds one
    instance and delegates to it.
    """

    __slots__ = (
        "_acquisition",
        "_thread_type",
        "_access_state",
        "_expected_ops",
        "_ror_count",
    )

    def __init__(self, acquisition: BlockAcquisition, thread_type: ThreadType) -> None:
        self._acquisition: BlockAcquisition = acquisition
        self._thread_type: ThreadType = thread_type
        self._access_state: AccessState = AccessState.OS
        self._expected_ops: Set[ExpectedOp] = set()
        self._ror_count: int = 0

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def acquisition(self) -> BlockAcquisition:
        return self._acquisition

    @property
    def thread_type(self) -> ThreadType:
        return self._thread_type

    @property
    def access_state(self) -> AccessState:
        return self._access_state

    @property
    def expected_ops(self) -> Set[ExpectedOp]:
        return self._expected_ops

    @property
    def ror_count(self) -> int:
        """Number of in-flight copies while in ROR state (0 when not in ROR)."""
        return self._ror_count

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set the initial state based on acquisition method and thread type."""
        if self._acquisition == BlockAcquisition.RESERVE:
            self._access_state = AccessState.MW
            if self._thread_type == ThreadType.DM:
                self._expected_ops = {ExpectedOp.COPY_DST}
            else:
                self._expected_ops = {ExpectedOp.STORE}
        elif self._acquisition == BlockAcquisition.WAIT:
            self._access_state = AccessState.MR
            if self._thread_type == ThreadType.DM:
                self._expected_ops = {ExpectedOp.COPY_SRC}
            else:
                self._expected_ops = {ExpectedOp.STORE_SRC}

    def set_unrestricted(self) -> None:
        """Set to RW with no expected-ops restrictions (used for temporary blocks)."""
        self._access_state = AccessState.RW
        self._expected_ops = set()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, operation: str, expected_op: ExpectedOp) -> None:
        """Raise RuntimeError if expected_op is not currently allowed.

        Args:
            operation: Human-readable operation name for error messages.
            expected_op: The operation being attempted.
        """
        if not self._expected_ops:
            raise RuntimeError(
                f"Cannot perform {operation}: Block is in DONE/uninitialized state. "
                f"No more operations are expected on this block. "
                f"Current state: {self._access_state.name}"
            )
        if expected_op not in self._expected_ops:
            expected_names = ", ".join(
                op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
            )
            raise RuntimeError(
                f"Cannot perform {operation}: Expected one of [{expected_names}], "
                f"but got {operation}. "
                f"Current state: Acquisition={self._acquisition.name}, "
                f"Thread={self._thread_type.name}, Access={self._access_state.name}"
            )

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        operation_key: str,
        operation_display: str,
        expected_op: ExpectedOp,
    ) -> None:
        """Execute a state-machine transition.

        Validates that expected_op is currently allowed, then applies the
        ROR(N) counter logic for copy_src / tx_wait while in ROR state, and
        falls through to the STATE_TRANSITIONS table for everything else.

        Args:
            operation_key: Table lookup key (e.g. "copy_src", "tx_wait").
            operation_display: Human-readable name used in error messages.
            expected_op: The operation being attempted (for validation).
        """
        self.validate(operation_display, expected_op)

        # ROR(N) in-state transitions: copy_src increments N; tx_wait
        # decrements N.  Only the final tx_wait (N == 1) falls through to the
        # table, which maps (tx_wait, ROR) -> RW.
        if self._access_state == AccessState.ROR:
            if operation_key == "copy_src":
                self._ror_count += 1
                self._expected_ops = _ROR_EXPECTED
                return
            if operation_key == "tx_wait" and self._ror_count > 1:
                self._ror_count -= 1
                self._expected_ops = _ROR_EXPECTED
                return

        context_key = (self._acquisition, self._thread_type)
        context_transitions = STATE_TRANSITIONS.get(context_key)

        if context_transitions is None:
            raise RuntimeError(
                f"Impossible! No transitions defined for: "
                f"Acquisition={self._acquisition.name}, "
                f"Thread={self._thread_type.name}"
            )

        transition_key = (operation_key, self._access_state)
        transition = context_transitions.get(transition_key)

        if transition is None:
            raise RuntimeError(
                f"Impossible! Invalid state for {operation_display}: "
                f"Acquisition={self._acquisition.name}, "
                f"Thread={self._thread_type.name}, "
                f"Access={self._access_state.name}"
            )

        new_access_state, new_expected_ops = transition
        self._access_state = new_access_state
        if new_access_state == AccessState.ROR:
            self._ror_count = 1
        self._expected_ops = new_expected_ops

    def transition_push(self) -> None:
        """Validate and execute the push() transition (RESERVE blocks only).

        Raises:
            RuntimeError: If PUSH is not expected, or if this is not a RESERVE block.
        """
        self.validate("push()", ExpectedOp.PUSH)
        if self._acquisition != BlockAcquisition.RESERVE:
            raise RuntimeError(
                f"Cannot perform push(): push() is only valid for reserve() blocks, "
                f"got {self._acquisition.name} block. "
                f"Current state: Thread={self._thread_type.name}, "
                f"Access={self._access_state.name}"
            )
        self._access_state = AccessState.OS
        self._expected_ops = set()

    def transition_assign_src(self) -> None:
        """Fire the assign_src transition (WAIT/COMPUTE blocks only).

        Called when the block's data is used as an arithmetic operand (assigned
        to a temporary).  Unlocks POP so the context manager can exit, but
        registers the block as pending store confirmation: the block's data
        must eventually reach a store() call, which is validated at program
        termination via DataflowBuffer.validate_no_pending_blocks().
        """
        self.transition("assign_src", "assign_src", ExpectedOp.STORE_SRC)

    def transition_pop(self) -> None:
        """Validate and execute the pop() transition (WAIT blocks only).

        The block must be in MR, RW, or A state.

        Raises:
            RuntimeError: If POP is not expected, if this is not a WAIT block,
                or if the current access state is not MR / RW / A.
        """
        self.validate("pop()", ExpectedOp.POP)
        if self._acquisition != BlockAcquisition.WAIT:
            raise RuntimeError(
                f"Cannot perform pop(): pop() is only valid for wait() blocks, "
                f"got {self._acquisition.name} block. "
                f"Current state: Thread={self._thread_type.name}, "
                f"Access={self._access_state.name}"
            )
        if self._access_state not in (AccessState.MR, AccessState.RW):
            raise RuntimeError(
                f"Cannot perform pop(): Invalid access state {self._access_state.name}. "
                f"Expected MR (never used) or RW (used as source)."
            )
        self._access_state = AccessState.OS
        self._expected_ops = set()
