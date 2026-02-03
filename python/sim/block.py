# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Block and supporting Span for cbsim.
"""

import operator as _op
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Sequence, Union

from pydantic import validate_call

from .cbstate import CBSlot
from .ttnnsim import Tensor
from .typedefs import Index, Shape, Size, Span


# Global variable to track current thread type in cooperative scheduling
_current_thread_type: Optional["ThreadType"] = None


def _get_current_thread_type() -> "ThreadType":
    """Get the current thread type.

    Returns:
        ThreadType

    Raises:
        RuntimeError: If thread type is not set (not within a thread context)
    """
    if _current_thread_type is None:
        raise RuntimeError(
            "Thread context not set. Must be called within a kernel thread or after "
            "calling _set_current_thread_type()."
        )
    return _current_thread_type


def _set_current_thread_type(thread_type: Optional["ThreadType"]) -> None:
    """Set the current thread type.

    Args:
        thread_type: The thread type to set, or None to clear the context
    """
    global _current_thread_type
    _current_thread_type = thread_type


def _clear_current_thread_type() -> None:
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
    PUSH = auto()  # Expect cb.push()
    POP = auto()  # Expect cb.pop()
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


# Notice that get_read_ptr and get_write_ptr return a C++ pointer which does not
# necessarily make sense in a python context. So we need something that can
# access the elements of the cb (as a pointer would) from the position the
# pointer points. To hide needless index arithmetic, we also add the ability to
# wrap around. Notice also that it handles a list and a capacity, instead of a
# _CBState, a deliberate choice to make it closer in spirit to a pointer and
# minimizing the state that is exposed.
class Block:
    """A logically contiguous window into the ring, possibly wrapping.
    Provides list-like access to elements while respecting wrap-around.

    State Machine:
    The block maintains a state machine that validates correct usage patterns:
    - Tracks acquisition method (reserve vs wait)
    - Tracks current thread type (DM vs Compute)
    - Tracks access state (RO/WO/RW/NA)
    - Tracks expected next operation
    - Transitions to DONE state after final operation (push/pop)
    """

    __slots__ = (
        "_buf",
        "_capacity",
        "_span",
        "_shape",
        "_acquisition",
        "_thread_type",
        "_access_state",
        "_expected_ops",
        "_is_temporary",
        "_source_blocks",  # Track wait() blocks that contributed to this temporary block
    )

    # TODO: We can't do @validate_call here. There reason is that @validate_call actually
    #       copies the arguments to validate them and returns the copies to the decorated
    #       function. In our case, we don't want the copy of the list, we want to use the
    #       original list as is. This is a limitation of pydantic's validate_call, and
    #       perhaps a good reason to look for other frameworks that don't do that! (beartype?)
    # @validate_call
    def __init__(
        self,
        buf: List[CBSlot],
        capacity: Size,
        span: Span,
        shape: Shape,
        acquisition: BlockAcquisition,
        thread_type: ThreadType,
        is_temporary: bool = False,
    ):
        self._buf = buf
        self._capacity = capacity
        self._span = span
        self._shape = shape
        self._is_temporary = is_temporary
        self._source_blocks: List["Block"] = []  # Track source wait() blocks

        # State machine variables
        self._acquisition: BlockAcquisition = acquisition
        self._thread_type: ThreadType = thread_type
        self._access_state: AccessState = AccessState.OS
        self._expected_ops: set[ExpectedOp] = set()  # Empty set = not initialized

        # Initialize state based on acquisition method and thread type
        # Skip state machine for temporary blocks (computation results)
        if not is_temporary:
            self._initialize_state()
        else:
            # Temporary blocks have full read/write access, no state machine
            self._access_state = AccessState.RW
            self._expected_ops = set()  # No restrictions

    def _initialize_state(self) -> None:
        """Initialize the block state machine based on acquisition and thread type.

        This is called automatically from __init__.
        """
        # Set initial state based on acquisition method and thread type
        if self._acquisition == BlockAcquisition.RESERVE:
            if self._thread_type == ThreadType.DM:
                self._access_state = AccessState.MW
                self._expected_ops = {
                    ExpectedOp.COPY_DST
                }  # DM reserves to receive data
            elif self._thread_type == ThreadType.COMPUTE:
                # Compute threads start in MW (must-write) state
                # Can choose either store(acc=False) or store(acc=True)
                # Note: store(acc=True) will transition to A before reading
                self._access_state = AccessState.MW
                self._expected_ops = {ExpectedOp.STORE, ExpectedOp.STORE_ACC}
        elif self._acquisition == BlockAcquisition.WAIT:
            # wait() blocks have data already present
            # DM threads copy out the data first, compute threads can read it directly
            if self._thread_type == ThreadType.DM:
                self._access_state = AccessState.MR
                self._expected_ops = {
                    ExpectedOp.COPY_SRC
                }  # DM threads copy data out first
            elif self._thread_type == ThreadType.COMPUTE:
                # Compute threads: wait() blocks start in MR state
                # Must be used as source in at least one store operation before pop
                self._access_state = AccessState.MR
                self._expected_ops = {ExpectedOp.STORE_SRC}

    @classmethod
    def from_list(
        cls,
        tensors: List[Tensor],
        shape: Shape,
    ) -> "Block":
        """Create a temporary Block from a list of tensors (computation result).

        Temporary blocks are not backed by CB storage and don't support wrap-around.
        """
        return cls(
            buf=tensors,
            capacity=len(tensors),
            span=Span(0, len(tensors)),
            shape=shape,
            acquisition=BlockAcquisition.RESERVE,  # Temporary blocks use RESERVE semantics
            thread_type=ThreadType.COMPUTE,  # Temporary blocks are from compute operations
            is_temporary=True,
        )

    def _validate_state(self, operation: str, expected_op: ExpectedOp) -> None:
        """Validate that the current operation is allowed in the current state.

        Args:
            operation: Name of the operation being performed
            expected_op: The operation being performed

        Raises:
            RuntimeError: If the operation is not allowed in the current state
        """
        # Note: We don't check AccessState.NA here because NA is valid for internal
        # state transitions (like tx.wait()). NA only blocks user access via
        # _check_can_read() and _check_can_write().

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
                f"Cannot perform {operation}: Expected one of [{expected_names}], but got {operation}. "
                f"Current state: Acquisition={self._acquisition.name}, "
                f"Thread={self._thread_type.name}, Access={self._access_state.name}"
            )

    def mark_copy_as_source(self) -> None:
        """Mark that this block is being used as a copy source.

        Valid states (per state machine diagram):
        - wait() DM RO COPY_SRC -> Expect copy(blk, ...) - single copy only
        - reserve() DM RW COPY_SRC -> Expect copy(blk, ...) - can do multiple copies

        Note:
        - wait() blocks: Only ONE copy operation allowed per block retrieval
        - reserve() DM blocks: Multiple copy operations allowed (loop back to RW COPY_SRC)
        - Only DM thread can use blocks as copy sources

        Block remains readable but writes are blocked until tx.wait().
        """
        # Validate that COPY_SRC is expected in current state
        self._validate_state("copy (as source)", ExpectedOp.COPY_SRC)

        # Validate complete state based on acquisition type
        if self._acquisition == BlockAcquisition.WAIT:
            # wait() DM MR COPY_SRC
            if self._thread_type != ThreadType.DM:
                raise RuntimeError(
                    f"Invalid state for copy source: wait() blocks must be in DM thread, got {self._thread_type.name}"
                )
            if self._access_state != AccessState.MR:
                raise RuntimeError(
                    f"Invalid state for copy source: wait() blocks must be in MR state, got {self._access_state.name}"
                )
            # Transition: keep MR state, expect TX_WAIT
            self._expected_ops = {ExpectedOp.TX_WAIT}

        elif self._acquisition == BlockAcquisition.RESERVE:
            # reserve() DM RW COPY_SRC (or COPY_DST/PUSH in loop)
            if self._thread_type != ThreadType.DM:
                raise RuntimeError(
                    f"Invalid state for copy source: reserve() blocks must be in DM thread, got {self._thread_type.name}"
                )
            if self._access_state != AccessState.RW:
                raise RuntimeError(
                    f"Invalid state for copy source: reserve() DM blocks must be in RW state, got {self._access_state.name}"
                )
            # Transition: RW -> NAR, expect TX_WAIT
            self._access_state = AccessState.NAR
            self._expected_ops = {ExpectedOp.TX_WAIT}
        else:
            raise RuntimeError(
                f"Invalid acquisition type for copy source: {self._acquisition.name}"
            )

    def mark_copy_as_dest(self) -> None:
        """Mark that this block is being used as a copy destination.

        Valid states (per state machine diagram):
        - reserve() DM WO COPY_DST -> First copy destination (initial state)
        - reserve() DM RW COPY_DST -> Can also be copy destination (after tx.wait() from previous copy)

        Note:
        - wait() blocks cannot be copy destinations per state machine
        - reserve() Compute blocks expect STORE, not copy as destination
        - Only DM thread can use blocks as copy destinations
        - reserve() DM blocks can do multiple copy operations (loop back to RW after tx.wait())

        This is called when a CopyTransaction is created to transition the block to NA state.
        """
        # Validate that COPY_DST is expected in current state
        self._validate_state("copy (as destination)", ExpectedOp.COPY_DST)

        # Validate complete state - only reserve() DM blocks can be copy destinations
        if self._acquisition != BlockAcquisition.RESERVE:
            raise RuntimeError(
                f"Invalid acquisition for copy destination: Expected RESERVE, got {self._acquisition.name}"
            )
        if self._thread_type != ThreadType.DM:
            raise RuntimeError(
                f"Invalid thread type for copy destination: Expected DM, got {self._thread_type.name}"
            )
        # Access state can be MW (initial) or RW (after tx.wait() in loop)
        if self._access_state not in (AccessState.MW, AccessState.RW):
            raise RuntimeError(
                f"Invalid access state for copy destination: Expected MW or RW, got {self._access_state.name}"
            )

        # After copy, block becomes NAW (user cannot access) and expects TX_WAIT
        self._access_state = AccessState.NAW
        self._expected_ops = {ExpectedOp.TX_WAIT}

    def mark_tx_wait_complete(self) -> None:
        """Mark that tx.wait() has completed for a copy operation.

        Valid states (per state machine diagram):
        - wait() DM RO TX_WAIT -> After copy as source, goes to RO POP
        - reserve() DM NA TX_WAIT -> After copy as destination, goes to RW (can do more copies or push)
        - reserve() DM RO TX_WAIT -> After copy as source, goes to RW (can do more copies or push)

        Only DM thread performs copy operations, so only DM blocks should call tx.wait().

        Note: reserve() DM blocks can now do multiple copy operations (as source or dest)
        before finally pushing.
        """
        # Validate expected operation
        self._validate_state("tx.wait()", ExpectedOp.TX_WAIT)

        # Validate complete state and transition based on acquisition type
        if self._acquisition == BlockAcquisition.RESERVE:
            # reserve() DM NAW/NAR TX_WAIT
            if self._thread_type != ThreadType.DM:
                raise RuntimeError(
                    f"Invalid thread type for tx.wait(): Expected DM for reserve() blocks, got {self._thread_type.name}"
                )
            # Can be NAW (after copy dest) or NAR (after copy src)
            if self._access_state not in (AccessState.NAW, AccessState.NAR):
                raise RuntimeError(
                    f"Invalid access state for tx.wait(): Expected NAW or NAR for reserve() blocks, got {self._access_state.name}"
                )
            # Transition: NAW/NAR -> RW (can do more copies or push)
            self._access_state = AccessState.RW
            self._expected_ops = {
                ExpectedOp.COPY_SRC,
                ExpectedOp.COPY_DST,
                ExpectedOp.PUSH,
            }

        elif self._acquisition == BlockAcquisition.WAIT:
            # wait() DM MR TX_WAIT
            if self._thread_type != ThreadType.DM:
                raise RuntimeError(
                    f"Invalid thread type for tx.wait(): Expected DM for wait() blocks, got {self._thread_type.name}"
                )
            if self._access_state != AccessState.MR:
                raise RuntimeError(
                    f"Invalid access state for tx.wait(): Expected MR for wait() blocks, got {self._access_state.name}"
                )
            # Transition: MR -> MR, expecting pop
            self._access_state = AccessState.MR
            self._expected_ops = {ExpectedOp.POP}
        else:
            raise RuntimeError(
                f"Invalid acquisition type for tx.wait(): {self._acquisition.name}"
            )

    def mark_store_read_complete(self) -> None:
        """Mark that this block was used as source (input) in a store operation.

        Valid states (per state machine diagram):
        - wait() Compute RO STORE_SRC -> Block used as source in store(), transitions to allow pop

        This is called when a wait() Compute block is used as the source argument
        in another block's store() call (e.g., output_block.store(input_block)).
        """
        # Validate that STORE_SRC is expected in current state
        self._validate_state("store (as source)", ExpectedOp.STORE_SRC)

        # Validate complete state
        if self._acquisition != BlockAcquisition.WAIT:
            raise RuntimeError(
                f"Invalid acquisition for store source: Expected WAIT, got {self._acquisition.name}"
            )
        if self._thread_type != ThreadType.COMPUTE:
            raise RuntimeError(
                f"Invalid thread type for store source: Expected COMPUTE, got {self._thread_type.name}"
            )
        # Can be MR (first use) or RW (subsequent uses as source)
        if self._access_state not in (AccessState.MR, AccessState.RW):
            raise RuntimeError(
                f"Invalid access state for store source: Expected MR or RW, got {self._access_state.name}"
            )

        # After being used as source in store, transition to RW (can be used in more stores or popped)
        self._access_state = AccessState.RW
        self._expected_ops = {ExpectedOp.STORE_SRC, ExpectedOp.POP}

    def mark_store_complete(self, acc: bool = False) -> None:
        """Mark that store() has completed on this block (as destination).

        Valid states (per state machine diagram):

        For reserve() Compute blocks:
        Two distinct paths from initial reserve() Compute WO {STORE, STORE_ACC}:

        Path 1 (single non-acc store):
          reserve() Compute WO {STORE, STORE_ACC} → store(acc=False) → reserve() Compute RO {PUSH}
          After single non-acc store, MUST push (no more stores allowed)

        Path 2 (multiple acc stores):
          reserve() Compute WO {STORE, STORE_ACC} → store(acc=True) → reserve() Compute RW {STORE_ACC, PUSH}
          Can continue with more store(acc=True) or push

        Note: wait() blocks are never destinations for store operations in compute threads.
        They are used as sources (see mark_store_read_complete).

        These paths cannot be mixed - once you choose one path, you must follow it.
        """
        if acc:
            # Accumulator store path - multiple stores allowed
            # First call expects STORE_ACC (from WO), subsequent calls also expect STORE_ACC (from RW)
            self._validate_state("store(acc=True)", ExpectedOp.STORE_ACC)

            # Validate complete state - only reserve() blocks can use acc=True
            if self._acquisition != BlockAcquisition.RESERVE:
                raise RuntimeError(
                    f"Invalid acquisition for store(acc=True): Expected RESERVE, got {self._acquisition.name}"
                )
            if self._thread_type != ThreadType.COMPUTE:
                raise RuntimeError(
                    f"Invalid thread type for store(acc=True): Expected COMPUTE, got {self._thread_type.name}"
                )
            # Can be MW (first store) or A (subsequent stores)
            if self._access_state not in (AccessState.MW, AccessState.A):
                raise RuntimeError(
                    f"Invalid access state for store(acc=True): Expected MW or A, got {self._access_state.name}"
                )

            # After acc store, transition to A and expect more acc stores (or push)
            self._access_state = AccessState.A
            self._expected_ops = {ExpectedOp.STORE_ACC, ExpectedOp.PUSH}
        else:
            # Regular non-acc store - only for reserve() blocks
            self._validate_state("store()", ExpectedOp.STORE)

            # Validate complete state - only reserve() blocks are store destinations
            if self._acquisition != BlockAcquisition.RESERVE:
                raise RuntimeError(
                    f"Invalid acquisition for store(): Expected RESERVE, got {self._acquisition.name}"
                )
            if self._thread_type != ThreadType.COMPUTE:
                raise RuntimeError(
                    f"Invalid thread type for store(): Expected COMPUTE, got {self._thread_type.name}"
                )
            if self._access_state != AccessState.MW:
                raise RuntimeError(
                    f"Invalid access state for store(): Expected MW, got {self._access_state.name}"
                )

            # After non-acc store, transition to MR and expect push
            self._access_state = AccessState.MR
            self._expected_ops = {ExpectedOp.PUSH}

    def mark_push_complete(self) -> None:
        """Mark that push() has completed.

        Valid states (per state machine diagram):
        - Must be a reserve() block (not wait())
        - Must have PUSH in expected operations
        """
        # Validate acquisition
        if self._acquisition != BlockAcquisition.RESERVE:
            raise RuntimeError(
                f"Cannot perform push(): Expected RESERVE acquisition, got {self._acquisition.name}. "
                f"Current state: Thread={self._thread_type.name}, Access={self._access_state.name}, "
                f"Expected Ops={{{', '.join(op.name for op in self._expected_ops)}}}"
            )

        # Validate that PUSH is in expected operations
        if ExpectedOp.PUSH not in self._expected_ops:
            expected_names = ", ".join(
                op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
            )
            raise RuntimeError(
                f"Cannot perform push(): Expected PUSH in expected operations, "
                f"but got {{{expected_names}}}. "
                f"Current state: Acquisition={self._acquisition.name}, "
                f"Thread={self._thread_type.name}, Access={self._access_state.name}"
            )

        # Transition to DONE (Out of Scope)
        self._access_state = AccessState.OS
        self._expected_ops = set()  # Empty = DONE

    def mark_pop_complete(self) -> None:
        """Mark that pop() has completed.

        Valid states (per state machine diagram):
        - wait() Compute RW POP -> Pop after being used as source in at least one store
        - wait() DM MR POP -> Pop after copy and tx.wait()
        """
        self._validate_state("pop()", ExpectedOp.POP)

        # Validate complete state
        if self._acquisition != BlockAcquisition.WAIT:
            raise RuntimeError(
                f"Invalid acquisition for pop(): Expected WAIT, got {self._acquisition.name}"
            )
        if self._access_state not in (AccessState.MR, AccessState.RW):
            raise RuntimeError(
                f"Invalid access state for pop(): Expected MR or RW, got {self._access_state.name}"
            )
        # Thread type can be either DM or COMPUTE for wait() blocks

        # After pop, block is done (Out of Scope)
        self._access_state = AccessState.OS
        self._expected_ops = set()  # Empty = DONE

    def __len__(self) -> Size:
        return self._span.length

    @property
    def is_temporary(self) -> bool:
        """Check if this Block is a temporary computation result (not CB-backed)."""
        return self._is_temporary

    def _check_can_read(self) -> None:
        """Check if this Block can be read from.

        Raises:
            RuntimeError: If state machine prohibits reading
        """
        # Temporary blocks can always be read
        if self._is_temporary:
            return

        # State machine check
        if self._access_state == AccessState.MW:
            expected_ops_str = (
                ", ".join(
                    op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
                )
                if self._expected_ops
                else "DONE"
            )
            raise RuntimeError(
                f"Cannot read from Block: Block is in must-write (MW) state. "
                f"Current state: {self._access_state.name}, Expected operations: [{expected_ops_str}]"
            )
        if self._access_state in (AccessState.NAW, AccessState.OS):
            expected_ops_str = (
                ", ".join(
                    op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
                )
                if self._expected_ops
                else "DONE"
            )
            raise RuntimeError(
                f"Cannot read from Block: Block has no access ({self._access_state.name} state). "
                f"Current state: {self._access_state.name}, Expected operations: [{expected_ops_str}]"
            )
        # NAR state (async read in progress) allows reads since we're copying FROM this block

    def _check_can_write(self) -> None:
        """Check if this Block can be written to.

        Raises:
            RuntimeError: If state machine prohibits writing
        """
        # Temporary blocks can always be written to
        if self._is_temporary:
            return

        # State machine check
        if self._access_state == AccessState.NAW:
            # NAW: Block is locked as copy destination until tx.wait() completes
            raise RuntimeError(
                f"Cannot write to Block: Block is locked as copy destination until tx.wait() completes (copy lock error). "
                f"Current state: {self._access_state.name}, Expected operations: [TX_WAIT]"
            )
        if self._access_state in (AccessState.NAR, AccessState.OS):
            expected_ops_str = (
                ", ".join(
                    op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
                )
                if self._expected_ops
                else "DONE"
            )
            raise RuntimeError(
                f"Cannot write to Block: Block has no access ({self._access_state.name} state). "
                f"Current state: {self._access_state.name}, Expected operations: [{expected_ops_str}]"
            )
        # Block writes during copy operations (when expecting TX_WAIT)
        # This applies to copy sources in NAR state
        if (
            ExpectedOp.TX_WAIT in self._expected_ops
            and self._access_state == AccessState.NAR
        ):
            raise RuntimeError(
                f"Cannot write to Block: Block is locked as copy source until tx.wait() completes. "
                f"Current state: {self._access_state.name}, Expected operations: [TX_WAIT]"
            )
        # Note: We allow writing in MR/RW/A states as appropriate for the operation

    @validate_call
    def __getitem__(self, idx: Index) -> Tensor:
        """Get item with lock checking."""
        self._check_can_read()
        if not (0 <= idx < self._span.length):
            raise IndexError(idx)

        # Temporary blocks don't wrap around
        if self._is_temporary:
            value = self._buf[idx]
        else:
            value = self._buf[(self._span.start + idx) % self._capacity]

        if value is None:
            raise ValueError(f"Reading uninitialized or consumed slot at index {idx}")
        return value

    # TODO: Why does validate_call fail here? Maybe because Tensor could
    # resolve to tensor which is similar to a list?
    # @validate_call
    def __setitem__(self, idx: Index, value: Tensor) -> None:
        """Direct assignment to Block is not allowed. Use store() or copy() instead."""
        raise RuntimeError(
            "Direct assignment to Block is not allowed. Use block.store() or copy() instead."
        )

    def _write_slot(self, idx: Index, value: Tensor) -> None:
        """Internal method to write to a slot. Only used by store() and copy handlers."""
        self._check_can_write()
        if not (0 <= idx < self._span.length):
            raise IndexError(idx)

        # Temporary blocks don't wrap around
        if self._is_temporary:
            self._buf[idx] = value
        else:
            self._buf[(self._span.start + idx) % self._capacity] = value

    @validate_call
    def pop(self, idx: Index) -> None:
        if not (0 <= idx < self._span.length):
            raise IndexError(idx)
        value = self._buf[(self._span.start + idx) % self._capacity]
        if value is None:
            raise ValueError(f"Popping uninitialized or consumed slot at index {idx}")
        self._buf[(self._span.start + idx) % self._capacity] = None

    def to_list(self) -> List[CBSlot]:
        return [self[i] for i in range(len(self))]

    def copy_as_dest(self, items: Sequence[Tensor]) -> None:
        """Store items into the block as part of a copy operation.

        This method is used by copy handlers and does NOT update the state machine.
        State transitions for copy operations are handled by mark_copy_as_dest()
        (called when CopyTransaction is created) and mark_tx_wait_complete().

        Args:
            items: Sequence of tensors to store
        """
        if len(items) != self._span.length:
            raise ValueError("Length mismatch in copy_as_dest()")

        # Store data without state checks - block is in NA state during copy
        # which is correct for the state machine, but we need to write the data
        for i, v in enumerate(items):
            self._buf[(self._span.start + i) % self._capacity] = v

    @staticmethod
    def _infer_broadcast_shape(left_shape: Shape, right_shape: Shape) -> Shape:
        """Infer the result shape from broadcasting two shapes.

        Uses standard broadcasting rules: dimensions must match or one must be 1.
        """
        if len(left_shape) != len(right_shape):
            # For now, require same number of dimensions
            raise ValueError(f"Shape dimension mismatch: {left_shape} vs {right_shape}")

        result_shape = tuple(
            max(l, r) if l == 1 or r == 1 or l == r else None
            for l, r in zip(left_shape, right_shape)
        )

        if None in result_shape:
            raise ValueError(
                f"Incompatible shapes for broadcasting: {left_shape} and {right_shape}"
            )

        return result_shape

    # @validate_call
    def store(self, items: Union["Block", Sequence[Tensor]], acc: bool = False) -> None:
        """Store items into the block.

        Args:
            items: Block or sequence of tensors to store
            acc: If True, accumulate with existing values (+=), otherwise assign (=)
                 Note: First store(acc=True) does assignment (y=x), subsequent ones accumulate (y+=x)
        """
        # Convert Block to sequence if needed, and track source blocks
        source_blocks_to_mark: List["Block"] = []
        if isinstance(items, Block):
            items_seq = items.to_list()
            # Check if this is a wait() Compute block being stored directly
            if (
                items._acquisition == BlockAcquisition.WAIT
                and items._thread_type == ThreadType.COMPUTE
                and ExpectedOp.STORE_SRC in items._expected_ops
            ):
                source_blocks_to_mark.append(items)
            # Check if this is a temporary block with tracked source wait() blocks
            elif items._is_temporary and items._source_blocks:
                source_blocks_to_mark.extend(
                    blk
                    for blk in items._source_blocks
                    if ExpectedOp.STORE_SRC in blk._expected_ops
                )
        else:
            items_seq = items

        if len(items_seq) != self._span.length:
            raise ValueError("Length mismatch in store()")

        # Check write access first (provides better error message for NA state)
        self._check_can_write()

        # Mark all wait() Compute source blocks as used
        for source_block in source_blocks_to_mark:
            source_block.mark_store_read_complete()

        # Determine if this is the first store(acc=True) by checking if we're in MW state
        is_first_acc_store = acc and self._access_state == AccessState.MW

        # Mark state machine transition BEFORE actual store (needed for acc=True to read)
        self.mark_store_complete(acc=acc)

        if acc:
            if is_first_acc_store:
                # First store(acc=True): Just assign (y = x), don't accumulate
                for i, v in enumerate(items_seq):
                    self._write_slot(i, v)
            else:
                # Subsequent store(acc=True): Accumulate (y += x)
                for i, v in enumerate(items_seq):
                    self._write_slot(i, self[i] + v)
        else:
            # Regular assignment
            for i, v in enumerate(items_seq):
                self._write_slot(i, v)

    def _apply_binary_op(
        self,
        left: "Block",
        right: "Block",
        op: Callable[[Any, Any], Any],
    ) -> List[Tensor]:
        """Element-wise binary op: left (op) right.

        Supports NumPy-style implicit broadcasting when shapes are compatible.
        """
        len_left = len(left)
        len_right = len(right)
        left_shape = left._shape
        right_shape = right._shape

        # Check if shapes match exactly - fast path
        if left_shape == right_shape and len_left == len_right:
            return [op(left[i], right[i]) for i in range(len_left)]

        # Check if broadcasting is valid using standard broadcasting rules
        # For now, require same number of dimensions
        if len(left_shape) != len(right_shape):
            raise ValueError(
                f"Cannot broadcast: dimension mismatch between shapes {left_shape} and {right_shape}. "
                f"Shapes must have the same number of dimensions."
            )

        # Check each dimension is compatible for broadcasting
        # Compatible means: equal, or one of them is 1
        for i, (l_dim, r_dim) in enumerate(zip(left_shape, right_shape)):
            if l_dim != r_dim and l_dim != 1 and r_dim != 1:
                raise ValueError(
                    f"Cannot broadcast: incompatible shapes {left_shape} and {right_shape}. "
                    f"Dimension {i} has sizes {l_dim} and {r_dim} which are incompatible "
                    f"(must be equal or one must be 1)."
                )

        # Shapes are compatible - perform the operation with broadcasting
        from .ttnnsim import broadcast_tensors

        # Convert to list and ensure all slots are Tensors (not None)
        left_list = left.to_list()
        right_list = right.to_list()

        # Type cast to assert these are Tensors (they should be, as blocks with data should have no None slots)
        left_tensors: List[Tensor] = left_list  # type: ignore[assignment]
        right_tensors: List[Tensor] = right_list  # type: ignore[assignment]

        return broadcast_tensors(
            left_tensors, right_tensors, left_shape, right_shape, op
        )

    def _binary_op(
        self,
        other: "Block",
        op: Callable[[Any, Any], Any],
    ) -> "Block":
        """Element-wise binary op: self (op) other.

        Tracks wait() Compute blocks that contribute to the result.
        """
        result_list = self._apply_binary_op(self, other, op)

        # Infer result shape using broadcasting rules
        result_shape = self._infer_broadcast_shape(self._shape, other._shape)

        result_block = Block.from_list(result_list, result_shape)

        # Track source wait() blocks that contributed to this result
        for block in [self, other]:
            # Unwrap if this is a context manager wrapper (WaitContext, ReserveContext)
            actual_block = block
            if hasattr(block, "_block"):
                actual_block = block._block  # type: ignore[attr-defined]

            # Only track if this is actually a Block object
            if not isinstance(actual_block, Block):
                continue

            if (
                not actual_block._is_temporary
                and actual_block._acquisition == BlockAcquisition.WAIT
                and actual_block._thread_type == ThreadType.COMPUTE
            ):
                result_block._source_blocks.append(actual_block)
            elif actual_block._is_temporary:
                # Temporary blocks may have their own source blocks to propagate
                result_block._source_blocks.extend(actual_block._source_blocks)

        return result_block

    # ---- forward operators ----

    def __add__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.add)

    def __sub__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.sub)

    def __mul__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.mul)

    def __truediv__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.truediv)

    def __floordiv__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.floordiv)

    def __mod__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.mod)

    def __pow__(self, other: Union["Block", int]) -> "Block":
        """Element-wise exponentiation.

        Supports both Block and scalar integer exponents.
        """
        if isinstance(other, int):
            # Scalar power - apply to each tensor in the block
            result_tensors = [t**other for t in self.to_list()]
            result_block = Block.from_list(result_tensors, shape=self._shape)

            # Track source wait() blocks
            # Unwrap if this is a context manager wrapper
            actual_self = self
            if hasattr(self, "_block"):
                actual_self = self._block  # type: ignore[attr-defined]

            if isinstance(actual_self, Block):
                if (
                    not actual_self._is_temporary
                    and actual_self._acquisition == BlockAcquisition.WAIT
                    and actual_self._thread_type == ThreadType.COMPUTE
                ):
                    result_block._source_blocks.append(actual_self)
                elif actual_self._is_temporary:
                    result_block._source_blocks.extend(actual_self._source_blocks)

            return result_block

        # Block power
        return self._binary_op(other, _op.pow)

    def __matmul__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.matmul)

    @property
    def shape(self) -> Shape:
        """Get the shape (rows, cols in tiles) of this block from its associated CB."""
        return self._shape


# Export thread context functions for use by program scheduler
__all__ = [
    "Block",
    "AccessState",
    "ThreadType",
    "BlockAcquisition",
    "ExpectedOp",
    "_get_current_thread_type",
    "_set_current_thread_type",
    "_clear_current_thread_type",
]
