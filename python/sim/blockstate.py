# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Block state machine enumerations and transition table.

Defines the kernel-type context, access-state machine, and the full
transition table used by Block to validate correct usage patterns.
"""

from enum import IntEnum, auto
from typing import AbstractSet, Callable, Dict, FrozenSet, Iterable, Optional, Tuple

# Type alias for the lazy callsite used in error messages.  Block-level access
# transitions are on the simulator's hottest path; passing a callable instead
# of an already-resolved ``(file, line)`` tuple lets ``validate()`` skip the
# state-machine bookkeeping for ``_pending_copy_site_for_errors`` entirely on
# the happy path -- the lookup only runs when an error is about to be raised.
PendingCopyLocationProvider = Callable[[], Optional[Tuple[str, int]]]


# All enums below are ``IntEnum`` rather than plain ``Enum`` so that
# ``__hash__`` falls through to ``int.__hash__`` (a single C-level op) instead
# of going through the much slower ``enum.Enum.__hash__`` defined in CPython's
# ``enum`` module.  These enums are used as dict/set keys on the hot
# block-state-machine and scheduler paths; profiling step_1 showed
# ``enum.__hash__`` accounting for ~4% of total runtime, which this change
# eliminates.  The user-visible behavior is unchanged: ``IntEnum`` is still
# an ``Enum`` (passes ``isinstance(x, Enum)``) and supports the same identity
# semantics; it only adds equality with the underlying integer, which the
# simulator does not rely on.
class AccessState(IntEnum):
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


class KernelType(IntEnum):
    """Kernel role for block operations (compute vs datamovement)."""

    DM = auto()  # Data Movement
    COMPUTE = auto()  # Compute


class BlockAcquisition(IntEnum):
    """How the block was acquired."""

    RESERVE = auto()  # Via reserve()
    WAIT = auto()  # Via wait()


class ExpectedOp(IntEnum):
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


# ------------------------------------------------------------------
# User-facing error message helpers
# ------------------------------------------------------------------


def _sorted_op_names(ops: Iterable[ExpectedOp]) -> str:
    return ", ".join(op.name for op in sorted(ops, key=lambda x: x.name))


# Short "next op" hints (per ExpectedOp), appended after "Next:" in mismatch errors.
_EXPECTED_OP_GUIDANCE: Dict[ExpectedOp, str] = {
    ExpectedOp.COPY_SRC: "copy(block, dest_tensor) with this block as the source",
    ExpectedOp.COPY_DST: "copy(src, block) with this block as the destination",
    ExpectedOp.TX_WAIT: "wait until the copy on this block completes",
    ExpectedOp.PUSH: "push() when the reserve() buffer is written and the producer is done",
    ExpectedOp.POP: "pop() when the wait() buffer is no longer needed",
    ExpectedOp.STORE: "block.store(…) as destination (compute path)",
    ExpectedOp.STORE_SRC: "out_block.store(this_block, …) with this block as the source operand",
    ExpectedOp.DONE: "none (block finished)",
}


def _guidance_for_expected_ops(ops: AbstractSet[ExpectedOp]) -> str:
    parts = [
        _EXPECTED_OP_GUIDANCE[o]
        for o in sorted(ops, key=lambda x: x.name)
        if o in _EXPECTED_OP_GUIDANCE
    ]
    if not parts:
        return "see dataflow block contract in docs"
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts)


def _validate_mismatch_hint(
    attempted: ExpectedOp,
    expected_ops: AbstractSet[ExpectedOp],
    access: AccessState,
    acquisition: BlockAcquisition,
    kernel: KernelType,
) -> Optional[str]:
    """What the mistake usually means; None selects the generic secondary sentence."""
    if attempted == ExpectedOp.PUSH and acquisition == BlockAcquisition.WAIT:
        return "push() is for reserve() only; a wait() block is closed with pop()."
    if attempted == ExpectedOp.POP and acquisition == BlockAcquisition.RESERVE:
        return "pop() is for wait() only; a reserve() block is closed with push()."
    if acquisition == BlockAcquisition.WAIT and access in (
        AccessState.MR,
        AccessState.RW,
    ):
        if kernel == KernelType.DM:
            if attempted == ExpectedOp.COPY_DST and ExpectedOp.COPY_SRC in expected_ops:
                return (
                    "After wait(), data is already in the block: copy *from* it first, not into it (unless the "
                    "state machine already allows a destination copy)."
                )
        if kernel == KernelType.COMPUTE:
            if attempted == ExpectedOp.STORE and ExpectedOp.STORE_SRC in expected_ops:
                return (
                    "A wait() block is not written in place with store(...); pass this block as the source to "
                    "another block's store (out_block.store(this_block, …))."
                )
    if access == AccessState.NAW:
        return "A copy may still be in flight (NAW); wait for it to finish before other uses."
    if access == AccessState.ROR and attempted in (
        ExpectedOp.COPY_DST,
        ExpectedOp.STORE,
    ):
        return (
            "This block is a copy source while other copies may still read from it (ROR); "
            "wait for those copies to finish before writing into it."
        )
    if access == AccessState.MW and acquisition == BlockAcquisition.RESERVE:
        if attempted == ExpectedOp.COPY_SRC:
            return "reserve() view is still empty: copy or store into it before using it as a copy source."
    return None


def format_validate_mismatch(
    operation: str,
    attempted: ExpectedOp,
    expected_ops: AbstractSet[ExpectedOp],
    access: AccessState,
    acquisition: BlockAcquisition,
    kernel: KernelType,
    pending_copy_location: Optional[Tuple[str, int]] = None,
) -> str:
    expected_names = _sorted_op_names(expected_ops)
    hint = _validate_mismatch_hint(attempted, expected_ops, access, acquisition, kernel)
    follow = _guidance_for_expected_ops(expected_ops)
    body = [
        f"Cannot perform {operation}: not a valid next dataflow step for this block.",
    ]
    if hint:
        body.append(hint)
    else:
        body.append(
            "Call does not match the next allowed op in the producer/consumer order."
        )
    if pending_copy_location is not None and access in (
        AccessState.NAW,
        AccessState.ROR,
    ):
        path, line = pending_copy_location
        body.append(
            f"Where: the copy involving this block was requested at {path}:{line}."
        )
    body.append(f"Next: {follow}.")
    body.append(
        f"Details: expected one of [{expected_names}], attempted {attempted.name}, "
        f"acquisition={acquisition.name}, kernel={kernel.name}, access={access.name}."
    )
    return "\n\n".join(body)


def format_block_finished_error(operation: str, access: AccessState) -> str:
    return (
        f"Cannot perform {operation}: block is no longer active (push/pop already, or not initialized here).\n\n"
        f"Next: new block from reserve() or wait(); do not reuse after push()/pop().\n\n"
        f"Details: access={access.name}, expected-ops=empty (DONE)."
    )


def _read_lead(name: Optional[str]) -> str:
    """Opening clause so the user always sees which buffer block the error is about (optional label)."""
    if name and str(name).strip():
        return f"Cannot read from this buffer block (name: {name!r})"
    return "Cannot read from this buffer block"


def _write_lead(name: Optional[str]) -> str:
    if name and str(name).strip():
        return f"Cannot write to this buffer block (name: {name!r})"
    return "Cannot write to this buffer block"


def _pending_copy_where_line(
    access: AccessState,
    pending_copy_location: Optional[Tuple[str, int]],
) -> str:
    """Extra line when NAW/ROR and we recorded the user callsite of copy(...) involving this block."""
    if pending_copy_location is None:
        return ""
    path, line = pending_copy_location
    if access == AccessState.NAW:
        return f"\n\nWhere: copy into this block was requested at {path}:{line}."
    if access == AccessState.ROR:
        return f"\n\nWhere: copy from this block was requested at {path}:{line}."
    return ""


def format_cannot_read_block(
    access: AccessState,
    expected_ops: AbstractSet[ExpectedOp],
    acquisition: BlockAcquisition,
    block_name: Optional[str] = None,
    pending_copy_location: Optional[Tuple[str, int]] = None,
) -> str:
    lead = _read_lead(block_name)
    exp = _sorted_op_names(expected_ops) if expected_ops else "DONE"
    if access == AccessState.MW:
        return (
            f"{lead}: MW (must-write) — not loaded yet; copy or store into this block first.\n\n"
            f"Next: see allowed ops in Details.\n\n"
            f"Details: state=MW, next allowed [{exp}], acquisition={acquisition.name}."
        )
    if access == AccessState.NAW:
        where = _pending_copy_where_line(access, pending_copy_location)
        return (
            f"{lead}: NAW — a copy into this block may still be in flight; wait for that copy to complete before reading "
            f"(same constraint as the copy-destination write lock for writes on this block)."
            f"{where}\n\n"
            f"Details: state=NAW, next allowed [{exp}]."
        )
    if access == AccessState.OS:
        return (
            f"{lead}: OS — out of scope (not readable after the block is returned with push() or pop()).\n\n"
            f"Next: do not use this block handle again; get a new block for more work (reserve() or wait()) if needed."
            f"\n\n"
            f"Details: state=OS, next allowed [{exp}], acquisition={acquisition.name}."
        )
    return f"{lead}. Details: state={access.name}, next allowed [{exp}], acquisition={acquisition.name}."


def format_cannot_write_block(
    access: AccessState,
    expected_ops: AbstractSet[ExpectedOp],
    block_name: Optional[str] = None,
    pending_copy_location: Optional[Tuple[str, int]] = None,
) -> str:
    lead = _write_lead(block_name)
    exp = _sorted_op_names(expected_ops) if expected_ops else "DONE"
    if access == AccessState.NAW:
        where = _pending_copy_where_line(access, pending_copy_location)
        return (
            f"{lead}: NAW, copy-destination lock (copy lock error) — an in-flight copy is potentially still writing; "
            f"wait for that copy to complete before the next use."
            f"{where}\n\n"
            f"Next: then follow the next allowed op in Details.\n\n"
            f"Details: state=NAW, next allowed [{exp}]."
        )
    if access == AccessState.ROR:
        where = _pending_copy_where_line(access, pending_copy_location)
        return (
            f"{lead}: in ROR with in-flight copy-source uses; no overwrite until each in-flight copy that reads from "
            f"this block has completed (use the wait your copy API provides for each one)."
            f"{where}\n\n"
            f"Next: then follow the next allowed op in Details.\n\n"
            f"Details: state=ROR, next allowed [{exp}]."
        )
    if access == AccessState.OS:
        return (
            f"{lead}: OS — not writable; the block is out of scope (returned after push or pop on this DFB path)."
            f"\n\n"
            f"Next: not applicable; use a new block from reserve() or wait().\n\n"
            f"Details: state=OS, next allowed [{exp}]."
        )
    return f"{lead}. Details: state={access.name}, next allowed [{exp}]."


# Module-level frozensets for ``expected_ops`` -- assigning these
# immutable singletons rather than constructing fresh ``set()`` literals on
# every state machine init / transition avoids 16.9M short-lived ``set``
# allocations per matmul-tutorial dry run.  ``frozenset`` is safe to share
# across all Blocks because callers only read it (``in`` / iteration /
# truthy) and always replace the field rather than mutating it.
_EMPTY_OPS: FrozenSet[ExpectedOp] = frozenset()
_INIT_RESERVE_DM: FrozenSet[ExpectedOp] = frozenset({ExpectedOp.COPY_DST})
_INIT_RESERVE_COMPUTE: FrozenSet[ExpectedOp] = frozenset({ExpectedOp.STORE})
_INIT_WAIT_DM: FrozenSet[ExpectedOp] = frozenset({ExpectedOp.COPY_SRC})
_INIT_WAIT_COMPUTE: FrozenSet[ExpectedOp] = frozenset({ExpectedOp.STORE_SRC})

# Transition table values, also shared frozensets so a transition is just
# a dict lookup + reference assignment with zero allocation.
_OPS_TX_AND_COPY_SRC: FrozenSet[ExpectedOp] = frozenset(
    {ExpectedOp.TX_WAIT, ExpectedOp.COPY_SRC}
)
_OPS_TX: FrozenSet[ExpectedOp] = frozenset({ExpectedOp.TX_WAIT})
_OPS_COPY_DST_SRC_POP: FrozenSet[ExpectedOp] = frozenset(
    {ExpectedOp.COPY_DST, ExpectedOp.COPY_SRC, ExpectedOp.POP}
)
_OPS_COPY_SRC: FrozenSet[ExpectedOp] = frozenset({ExpectedOp.COPY_SRC})
_OPS_PUSH_AND_COPY_SRC: FrozenSet[ExpectedOp] = frozenset(
    {ExpectedOp.PUSH, ExpectedOp.COPY_SRC}
)
_OPS_COPY_DST_SRC_PUSH: FrozenSet[ExpectedOp] = frozenset(
    {ExpectedOp.COPY_DST, ExpectedOp.COPY_SRC, ExpectedOp.PUSH}
)
_OPS_STORE_RW_POP: FrozenSet[ExpectedOp] = frozenset(
    {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.POP}
)
_OPS_STORE_SRC: FrozenSet[ExpectedOp] = frozenset({ExpectedOp.STORE_SRC})
_OPS_STORE_RW_PUSH: FrozenSet[ExpectedOp] = frozenset(
    {ExpectedOp.STORE_SRC, ExpectedOp.STORE, ExpectedOp.PUSH}
)
_OPS_STORE_SRC_PUSH: FrozenSet[ExpectedOp] = frozenset(
    {ExpectedOp.STORE_SRC, ExpectedOp.PUSH}
)

# State machine transition table
# Organized by (acquisition, kernel_type) -> {(operation, access_state): (new_access_state, new_expected_ops)}
# This structure makes it easy to see all transitions for a particular acquisition/kernel-role combination
STATE_TRANSITIONS: Dict[
    Tuple[BlockAcquisition, KernelType],
    Dict[
        Tuple[str, AccessState],
        Tuple[AccessState, FrozenSet[ExpectedOp]],
    ],
] = {
    # DM kernel, WAIT acquisition
    (BlockAcquisition.WAIT, KernelType.DM): {
        # Copy as source: MR/RW -> ROR; further copies and tx_wait both expected
        ("copy_src", AccessState.MR): (AccessState.ROR, _OPS_TX_AND_COPY_SRC),
        ("copy_src", AccessState.RW): (AccessState.ROR, _OPS_TX_AND_COPY_SRC),
        # Copy as destination: RW -> NAW + TX_WAIT
        ("copy_dst", AccessState.RW): (AccessState.NAW, _OPS_TX),
        # TX wait complete from ROR (N==1) -> RW with copy + pop ops
        ("tx_wait", AccessState.ROR): (AccessState.RW, _OPS_COPY_DST_SRC_POP),
        # TX wait complete from NAW -> MR with copy_src only
        ("tx_wait", AccessState.NAW): (AccessState.MR, _OPS_COPY_SRC),
    },
    # DM kernel, RESERVE acquisition
    (BlockAcquisition.RESERVE, KernelType.DM): {
        # Copy as source: MR/RW -> ROR; further copies and tx_wait both expected
        ("copy_src", AccessState.MR): (AccessState.ROR, _OPS_TX_AND_COPY_SRC),
        ("copy_src", AccessState.RW): (AccessState.ROR, _OPS_TX_AND_COPY_SRC),
        # Copy as destination: MW/RW -> NAW + TX_WAIT
        ("copy_dst", AccessState.MW): (AccessState.NAW, _OPS_TX),
        ("copy_dst", AccessState.RW): (AccessState.NAW, _OPS_TX),
        # TX wait complete from NAW -> MR with push + copy_src
        ("tx_wait", AccessState.NAW): (AccessState.MR, _OPS_PUSH_AND_COPY_SRC),
        # TX wait complete from ROR (N==1) -> RW with all copy ops + push
        ("tx_wait", AccessState.ROR): (AccessState.RW, _OPS_COPY_DST_SRC_PUSH),
    },
    # COMPUTE kernel, WAIT acquisition
    (BlockAcquisition.WAIT, KernelType.COMPUTE): {
        # Assign as arithmetic source: MR/RW -> RW; POP now allowed but store
        # confirmation is deferred and tracked until program termination.
        ("assign_src", AccessState.MR): (AccessState.RW, _OPS_STORE_RW_POP),
        ("assign_src", AccessState.RW): (AccessState.RW, _OPS_STORE_RW_POP),
        # Store read complete: MR/RW -> RW with store ops + pop
        ("store_src", AccessState.MR): (AccessState.RW, _OPS_STORE_RW_POP),
        ("store_src", AccessState.RW): (AccessState.RW, _OPS_STORE_RW_POP),
        # Store complete: RW -> MR with store_src only
        ("store_dst", AccessState.RW): (AccessState.MR, _OPS_STORE_SRC),
    },
    # COMPUTE kernel, RESERVE acquisition
    (BlockAcquisition.RESERVE, KernelType.COMPUTE): {
        # Store read complete: MR/RW -> RW with store ops + push
        ("store_src", AccessState.MR): (AccessState.RW, _OPS_STORE_RW_PUSH),
        ("store_src", AccessState.RW): (AccessState.RW, _OPS_STORE_RW_PUSH),
        # Store complete: MW/RW -> MR with store_src + push
        ("store_dst", AccessState.MW): (AccessState.MR, _OPS_STORE_SRC_PUSH),
        ("store_dst", AccessState.RW): (AccessState.MR, _OPS_STORE_SRC_PUSH),
    },
}

# ROR expected-ops set, shared by all in-state ROR transitions.
_ROR_EXPECTED: FrozenSet[ExpectedOp] = _OPS_TX_AND_COPY_SRC


class BlockStateMachine:
    """All access-state logic for a Block: initial state, validation, and transitions.

    Owns the five state fields (acquisition, kernel_type, access_state, expected_ops,
    ror_count) and every method that mutates them.  Block in dfb.py holds one
    instance and delegates to it.
    """

    # The four publicly-readable fields below are plain slots rather than
    # ``@property``-wrapped privates because they are read on every Block
    # operation on the simulator's hottest path.  Replacing the property
    # descriptors with direct slot access avoids one CPython method call per
    # read; combined with the corresponding change in ``Block`` it eliminates
    # a two- or three-deep property chain that profiling showed cost several
    # seconds per matmul-tutorial dry run.  ``_ror_count`` stays private
    # because it is touched only by transition bookkeeping.
    __slots__ = (
        "acquisition",
        "kernel_type",
        "access_state",
        "expected_ops",
        "_ror_count",
        # Cached reference to STATE_TRANSITIONS[(acquisition, kernel_type)].
        # Looked up once in __init__ so the per-call ``transition()`` body
        # avoids one ``dict.get`` (23M+ removed lookups per matmul-tutorial
        # dry run) and one tuple allocation for the ``(acquisition,
        # kernel_type)`` key.
        "_ctx_transitions",
    )

    def __init__(self, acquisition: BlockAcquisition, kernel_type: KernelType) -> None:
        self.acquisition: BlockAcquisition = acquisition
        self.kernel_type: KernelType = kernel_type
        self.access_state: AccessState = AccessState.OS
        # Module-level shared frozenset; ``initialize()`` / transitions
        # overwrite this with another shared frozenset, so no per-instance
        # set allocation is ever made.
        self.expected_ops: AbstractSet[ExpectedOp] = _EMPTY_OPS
        self._ror_count: int = 0
        # Look up the per-(acquisition, kernel_type) transition table once
        # at construction so transition() avoids one dict.get + one tuple
        # allocation per call.  Surfacing the bug at construction (rather
        # than per-call) is fine: every Block uses one of the four
        # combinations the table covers.
        ctx = STATE_TRANSITIONS.get((acquisition, kernel_type))
        if ctx is None:
            raise RuntimeError(
                f"No state-machine table for this acquisition/kernel role (simulator bug).\n\n"
                f"Details: acquisition={acquisition.name}, kernel={kernel_type.name}."
            )
        self._ctx_transitions: Dict[
            Tuple[str, AccessState], Tuple[AccessState, FrozenSet[ExpectedOp]]
        ] = ctx

    @property
    def ror_count(self) -> int:
        """Number of in-flight copies while in ROR state (0 when not in ROR)."""
        return self._ror_count

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set the initial state based on acquisition method and kernel role."""
        if self.acquisition == BlockAcquisition.RESERVE:
            self.access_state = AccessState.MW
            if self.kernel_type == KernelType.DM:
                self.expected_ops = _INIT_RESERVE_DM
            else:
                self.expected_ops = _INIT_RESERVE_COMPUTE
        elif self.acquisition == BlockAcquisition.WAIT:
            self.access_state = AccessState.MR
            if self.kernel_type == KernelType.DM:
                self.expected_ops = _INIT_WAIT_DM
            else:
                self.expected_ops = _INIT_WAIT_COMPUTE

    def set_unrestricted(self) -> None:
        """Set to RW with no expected-ops restrictions (used for temporary blocks)."""
        self.access_state = AccessState.RW
        self.expected_ops = _EMPTY_OPS

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        operation: str,
        expected_op: ExpectedOp,
        pending_copy_location: Optional[PendingCopyLocationProvider] = None,
    ) -> None:
        """Raise RuntimeError if expected_op is not currently allowed.

        Note: the body is also inlined at the top of :meth:`transition`,
        :meth:`transition_push`, and :meth:`transition_pop` to avoid a
        function call on the simulator's hot path (~35M call frames per
        matmul-tutorial dry run).  This method is kept for external callers
        and tests; keep the inlined copies in sync with the logic here.

        Args:
            operation: Human-readable operation name for error messages.
            expected_op: The operation being attempted.
            pending_copy_location: Zero-arg callable that resolves the user
                ``(file, line)`` of the copy(...) involving this block while
                NAW/ROR, if known.  Invoked only on the error path so the
                happy path pays no resolution cost.
        """
        expected_ops = self.expected_ops
        if not expected_ops:
            raise RuntimeError(
                format_block_finished_error(operation, self.access_state)
            )
        if expected_op not in expected_ops:
            loc = pending_copy_location() if pending_copy_location else None
            raise RuntimeError(
                format_validate_mismatch(
                    operation,
                    expected_op,
                    expected_ops,
                    self.access_state,
                    self.acquisition,
                    self.kernel_type,
                    pending_copy_location=loc,
                )
            )

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def transition(
        self,
        operation_key: str,
        operation_display: str,
        expected_op: ExpectedOp,
        pending_copy_location: Optional[PendingCopyLocationProvider] = None,
    ) -> None:
        """Execute a state-machine transition.

        Validates that expected_op is currently allowed, then applies the
        ROR(N) counter logic for copy_src / tx_wait while in ROR state, and
        falls through to the STATE_TRANSITIONS table for everything else.

        Args:
            operation_key: Table lookup key (e.g. "copy_src", "tx_wait").
            operation_display: Human-readable name used in error messages.
            expected_op: The operation being attempted (for validation).
            pending_copy_location: Zero-arg callable resolving the user
                callsite for copy involving this block (NAW/ROR), if known.
                Forwarded to :meth:`validate` and only invoked on errors.
        """
        # Inlined ``validate()`` body: the simulator fires this method
        # ~23M times per matmul-tutorial dry run; folding the validation
        # avoids one function-call frame per transition.
        access_state = self.access_state
        expected_ops = self.expected_ops
        if not expected_ops:
            raise RuntimeError(
                format_block_finished_error(operation_display, access_state)
            )
        if expected_op not in expected_ops:
            loc = pending_copy_location() if pending_copy_location else None
            raise RuntimeError(
                format_validate_mismatch(
                    operation_display,
                    expected_op,
                    expected_ops,
                    access_state,
                    self.acquisition,
                    self.kernel_type,
                    pending_copy_location=loc,
                )
            )

        # ROR(N) in-state transitions: copy_src increments N; tx_wait
        # decrements N.  Only the final tx_wait (N == 1) falls through to the
        # table, which maps (tx_wait, ROR) -> RW.
        if access_state == AccessState.ROR:
            if operation_key == "copy_src":
                self._ror_count += 1
                self.expected_ops = _ROR_EXPECTED
                return
            if operation_key == "tx_wait" and self._ror_count > 1:
                self._ror_count -= 1
                self.expected_ops = _ROR_EXPECTED
                return

        # ``_ctx_transitions`` was cached at construction so we skip the
        # per-call STATE_TRANSITIONS lookup plus its (acquisition,
        # kernel_type) tuple allocation.
        transition = self._ctx_transitions.get((operation_key, access_state))
        if transition is None:
            raise RuntimeError(
                f"Invalid transition: {operation_display!r} in access={access_state.name} for "
                f"{self.acquisition.name}/{self.kernel_type.name} (internal inconsistency: validate() should have "
                f"failed first; file a repro).\n\n"
                f"Details: operation_key={operation_key!r}, access={access_state.name}."
            )

        new_access_state, new_expected_ops = transition
        self.access_state = new_access_state
        if new_access_state == AccessState.ROR:
            self._ror_count = 1
        self.expected_ops = new_expected_ops

    def transition_push(
        self,
        pending_copy_location: Optional[PendingCopyLocationProvider] = None,
    ) -> None:
        """Validate and execute the push() transition (RESERVE blocks only).

        Raises:
            RuntimeError: If PUSH is not expected, or if this is not a RESERVE block.
        """
        # Inlined ``validate()`` body; see :meth:`transition` for rationale.
        access_state = self.access_state
        expected_ops = self.expected_ops
        if not expected_ops:
            raise RuntimeError(format_block_finished_error("push()", access_state))
        if ExpectedOp.PUSH not in expected_ops:
            loc = pending_copy_location() if pending_copy_location else None
            raise RuntimeError(
                format_validate_mismatch(
                    "push()",
                    ExpectedOp.PUSH,
                    expected_ops,
                    access_state,
                    self.acquisition,
                    self.kernel_type,
                    pending_copy_location=loc,
                )
            )
        if self.acquisition != BlockAcquisition.RESERVE:
            raise RuntimeError(
                f"push() only for reserve() blocks; wait() blocks use pop() on the consumer.\n\n"
                f"Details: acquisition={self.acquisition.name}, kernel={self.kernel_type.name}, "
                f"access={access_state.name}."
            )
        self.access_state = AccessState.OS
        self.expected_ops = _EMPTY_OPS

    def transition_assign_src(self) -> None:
        """Fire the assign_src transition (WAIT/COMPUTE blocks only).

        Called when the block's data is used as an arithmetic operand (assigned
        to a temporary).  Unlocks POP so the context manager can exit, but
        registers the block as pending store confirmation: the block's data
        must eventually reach a store() call, which is validated at program
        termination via DataflowBuffer.validate_no_pending_blocks().
        """
        self.transition(
            "assign_src", "assign_src", ExpectedOp.STORE_SRC, pending_copy_location=None
        )

    def transition_pop(
        self,
        pending_copy_location: Optional[PendingCopyLocationProvider] = None,
    ) -> None:
        """Validate and execute the pop() transition (WAIT blocks only).

        The block must be in MR, RW, or A state.

        Raises:
            RuntimeError: If POP is not expected, if this is not a WAIT block,
                or if the current access state is not MR / RW / A.
        """
        # Inlined ``validate()`` body; see :meth:`transition` for rationale.
        access_state = self.access_state
        expected_ops = self.expected_ops
        if not expected_ops:
            raise RuntimeError(format_block_finished_error("pop()", access_state))
        if ExpectedOp.POP not in expected_ops:
            loc = pending_copy_location() if pending_copy_location else None
            raise RuntimeError(
                format_validate_mismatch(
                    "pop()",
                    ExpectedOp.POP,
                    expected_ops,
                    access_state,
                    self.acquisition,
                    self.kernel_type,
                    pending_copy_location=loc,
                )
            )
        if self.acquisition != BlockAcquisition.WAIT:
            raise RuntimeError(
                f"pop() only for wait() blocks; reserve() blocks use push() on the producer.\n\n"
                f"Details: acquisition={self.acquisition.name}, kernel={self.kernel_type.name}, "
                f"access={access_state.name}."
            )
        if access_state not in (AccessState.MR, AccessState.RW):
            raise RuntimeError(
                f"pop() only from MR or RW; current access is {access_state.name}.\n\n"
                f"Details: need MR (unused as source) or RW (read at least once)."
            )
        self.access_state = AccessState.OS
        self.expected_ops = _EMPTY_OPS
