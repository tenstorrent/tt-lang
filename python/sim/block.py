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


def _get_current_thread_type() -> Optional["ThreadType"]:
    """Get the current thread type.

    Returns:
        ThreadType if set, None otherwise
    """
    return _current_thread_type


def _set_current_thread_type(thread_type: "ThreadType") -> None:
    """Set the current thread type.

    Args:
        thread_type: The thread type to set
    """
    global _current_thread_type
    _current_thread_type = thread_type


def _clear_current_thread_type() -> None:
    """Clear the current thread type."""
    global _current_thread_type
    _current_thread_type = None


class AccessState(Enum):
    """Access state for a block in the state machine."""

    RO = auto()  # Read Only
    WO = auto()  # Write Only
    RW = auto()  # Read Write
    NA = auto()  # No Access


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
    STORE = auto()  # Expect blk.store(...) - regular store (acc=False)
    STORE_ACC = auto()  # Expect blk.store(..., acc=True) - accumulator store
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
        "_broadcast_dims",
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
        broadcast_dims: List[int] | None = None,
    ):
        self._buf = buf
        self._capacity = capacity
        self._span = span
        self._shape = shape
        self._is_temporary = is_temporary
        self._broadcast_dims = broadcast_dims or []

        # State machine variables
        self._acquisition: BlockAcquisition = acquisition
        self._thread_type: ThreadType = thread_type
        self._access_state: AccessState = AccessState.NA
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
                self._access_state = AccessState.WO
                self._expected_ops = {
                    ExpectedOp.COPY_DST
                }  # DM reserves to receive data
            elif self._thread_type == ThreadType.COMPUTE:
                # Compute threads start in WO (write-only) state
                # Can choose either store(acc=False) or store(acc=True)
                # Note: store(acc=True) will transition to RW before reading
                self._access_state = AccessState.WO
                self._expected_ops = {ExpectedOp.STORE, ExpectedOp.STORE_ACC}
        elif self._acquisition == BlockAcquisition.WAIT:
            # wait() blocks have data already present
            # DM threads copy out the data first, compute threads can read it directly
            if self._thread_type == ThreadType.DM:
                self._access_state = AccessState.RO
                self._expected_ops = {
                    ExpectedOp.COPY_SRC
                }  # DM threads copy data out first
            elif self._thread_type == ThreadType.COMPUTE:
                self._access_state = AccessState.RO
                self._expected_ops = {
                    ExpectedOp.POP
                }  # Compute threads can read then pop directly

    @classmethod
    def from_list(
        cls,
        tensors: List[Tensor],
        shape: Shape,
        broadcast_dims: List[int] | None = None,
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
            broadcast_dims=broadcast_dims,
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
            # wait() DM RO COPY_SRC
            if self._thread_type != ThreadType.DM:
                raise RuntimeError(
                    f"Invalid state for copy source: wait() blocks must be in DM thread, got {self._thread_type.name}"
                )
            if self._access_state != AccessState.RO:
                raise RuntimeError(
                    f"Invalid state for copy source: wait() blocks must be in RO state, got {self._access_state.name}"
                )
            # Transition: keep RO state, expect TX_WAIT
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
            # Transition: RW -> RO, expect TX_WAIT
            self._access_state = AccessState.RO
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
        # Access state can be WO (initial) or RW (after tx.wait() in loop)
        if self._access_state not in (AccessState.WO, AccessState.RW):
            raise RuntimeError(
                f"Invalid access state for copy destination: Expected WO or RW, got {self._access_state.name}"
            )

        # After copy, block becomes NA (user cannot access) and expects TX_WAIT
        self._access_state = AccessState.NA
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
            # reserve() DM NA/RO TX_WAIT
            if self._thread_type != ThreadType.DM:
                raise RuntimeError(
                    f"Invalid thread type for tx.wait(): Expected DM for reserve() blocks, got {self._thread_type.name}"
                )
            # Can be NA (after copy dest) or RO (after copy src)
            if self._access_state not in (AccessState.NA, AccessState.RO):
                raise RuntimeError(
                    f"Invalid access state for tx.wait(): Expected NA or RO for reserve() blocks, got {self._access_state.name}"
                )
            # Transition: NA/RO -> RW (can do more copies or push)
            self._access_state = AccessState.RW
            self._expected_ops = {
                ExpectedOp.COPY_SRC,
                ExpectedOp.COPY_DST,
                ExpectedOp.PUSH,
            }

        elif self._acquisition == BlockAcquisition.WAIT:
            # wait() DM RO TX_WAIT
            if self._thread_type != ThreadType.DM:
                raise RuntimeError(
                    f"Invalid thread type for tx.wait(): Expected DM for wait() blocks, got {self._thread_type.name}"
                )
            if self._access_state != AccessState.RO:
                raise RuntimeError(
                    f"Invalid access state for tx.wait(): Expected RO for wait() blocks, got {self._access_state.name}"
                )
            # Transition: RO -> RO, expecting pop
            self._access_state = AccessState.RO
            self._expected_ops = {ExpectedOp.POP}
        else:
            raise RuntimeError(
                f"Invalid acquisition type for tx.wait(): {self._acquisition.name}"
            )

    def mark_store_complete(self, acc: bool = False) -> None:
        """Mark that store() has completed.

        Valid states (per state machine diagram):
        Two distinct paths from initial reserve() Compute WO {STORE, STORE_ACC}:

        Path 1 (single non-acc store):
          reserve() Compute WO {STORE, STORE_ACC} → store(acc=False) → reserve() Compute RO {PUSH}
          After single non-acc store, MUST push (no more stores allowed)

        Path 2 (multiple acc stores):
          reserve() Compute WO {STORE, STORE_ACC} → store(acc=True) → reserve() Compute RW {STORE_ACC, PUSH}
          Can continue with more store(acc=True) or push

        These paths cannot be mixed - once you choose one path, you must follow it.
        """
        if acc:
            # Accumulator store path - multiple stores allowed
            # First call expects STORE_ACC (from WO), subsequent calls also expect STORE_ACC (from RW)
            self._validate_state("store(acc=True)", ExpectedOp.STORE_ACC)

            # Validate complete state
            if self._acquisition != BlockAcquisition.RESERVE:
                raise RuntimeError(
                    f"Invalid acquisition for store(acc=True): Expected RESERVE, got {self._acquisition.name}"
                )
            if self._thread_type != ThreadType.COMPUTE:
                raise RuntimeError(
                    f"Invalid thread type for store(acc=True): Expected COMPUTE, got {self._thread_type.name}"
                )
            # Can be WO (first store) or RW (subsequent stores)
            if self._access_state not in (AccessState.WO, AccessState.RW):
                raise RuntimeError(
                    f"Invalid access state for store(acc=True): Expected WO or RW, got {self._access_state.name}"
                )

            # After acc store, transition to RW and expect more acc stores (or push)
            self._access_state = AccessState.RW
            self._expected_ops = {ExpectedOp.STORE_ACC, ExpectedOp.PUSH}
        else:
            # Regular non-acc store - only ONE allowed, then must push
            self._validate_state("store()", ExpectedOp.STORE)

            # Validate complete state
            if self._acquisition != BlockAcquisition.RESERVE:
                raise RuntimeError(
                    f"Invalid acquisition for store(): Expected RESERVE, got {self._acquisition.name}"
                )
            if self._thread_type != ThreadType.COMPUTE:
                raise RuntimeError(
                    f"Invalid thread type for store(): Expected COMPUTE, got {self._thread_type.name}"
                )
            if self._access_state != AccessState.WO:
                raise RuntimeError(
                    f"Invalid access state for store(): Expected WO, got {self._access_state.name}"
                )

            # After non-acc store, transition to RO and expect push
            self._access_state = AccessState.RO
            self._expected_ops = {ExpectedOp.PUSH}

    def mark_push_complete(self) -> None:
        """Mark that push() has completed.

        Valid for reserve() blocks (both DM and Compute).
        Can be called after STORE, STORE_ACC, COPY operations, or from RW COPY_SRC loop state.
        This is important for error handling - if an exception occurs before the expected
        operation completes, the context manager still calls push() on exit.

        We allow push from ANY expected operation for error handling purposes.
        """
        # For error handling, allow push from any state (don't validate expected_ops)
        # Just transition to DONE
        self._access_state = AccessState.NA
        self._expected_ops = set()  # Empty = DONE

    def mark_pop_complete(self) -> None:
        """Mark that pop() has completed.

        Valid states (per state machine diagram):
        - wait() Compute RO POP -> Direct pop after wait
        - wait() DM RO POP -> Pop after copy and tx.wait()
        """
        self._validate_state("pop()", ExpectedOp.POP)

        # Validate complete state
        if self._acquisition != BlockAcquisition.WAIT:
            raise RuntimeError(
                f"Invalid acquisition for pop(): Expected WAIT, got {self._acquisition.name}"
            )
        if self._access_state != AccessState.RO:
            raise RuntimeError(
                f"Invalid access state for pop(): Expected RO, got {self._access_state.name}"
            )
        # Thread type can be either DM or COMPUTE for wait() blocks

        # After pop, block is done
        self._access_state = AccessState.NA
        self._expected_ops = set()  # Empty = DONE

    def __len__(self) -> Size:
        return self._span.length

    @property
    def is_temporary(self) -> bool:
        """Check if this Block is a temporary computation result (not CB-backed)."""
        return self._is_temporary

    @property
    def broadcast_dims(self) -> List[int]:
        """Get the dimensions along which this block is marked for broadcasting."""
        return self._broadcast_dims

    def _check_can_read(self) -> None:
        """Check if this Block can be read from.

        Raises:
            RuntimeError: If state machine prohibits reading
        """
        # Temporary blocks can always be read
        if self._is_temporary:
            return

        # State machine check
        if self._access_state == AccessState.WO:
            expected_ops_str = (
                ", ".join(
                    op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
                )
                if self._expected_ops
                else "DONE"
            )
            raise RuntimeError(
                f"Cannot read from Block: Block is in write-only (WO) state. "
                f"Current state: {self._access_state.name}, Expected operations: [{expected_ops_str}]"
            )
        if self._access_state == AccessState.NA:
            expected_ops_str = (
                ", ".join(
                    op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
                )
                if self._expected_ops
                else "DONE"
            )
            raise RuntimeError(
                f"Cannot read from Block: Block has no access (NA state). "
                f"Current state: {self._access_state.name}, Expected operations: [{expected_ops_str}]"
            )

    def _check_can_write(self) -> None:
        """Check if this Block can be written to.

        Raises:
            RuntimeError: If state machine prohibits writing
        """
        # Temporary blocks can always be written to
        if self._is_temporary:
            return

        # State machine check
        if self._access_state == AccessState.NA:
            expected_ops_str = (
                ", ".join(
                    op.name for op in sorted(self._expected_ops, key=lambda x: x.name)
                )
                if self._expected_ops
                else "DONE"
            )
            raise RuntimeError(
                f"Cannot write to Block: Block has no access (NA state). "
                f"Current state: {self._access_state.name}, Expected operations: [{expected_ops_str}]"
            )
        # Block writes during copy operations (when expecting TX_WAIT)
        # This applies to both copy sources (RO, TX_WAIT) and destinations (NA, TX_WAIT)
        if (
            ExpectedOp.TX_WAIT in self._expected_ops
            and self._access_state == AccessState.RO
        ):
            raise RuntimeError(
                f"Cannot write to Block: Block is locked as copy source until tx.wait() completes. "
                f"Current state: {self._access_state.name}, Expected operations: [TX_WAIT]"
            )
        # Note: We allow writing in RO state for reserve() blocks that can call store() multiple times
        # (when expected_op is STORE or PUSH, not TX_WAIT)

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
        # Convert Block to sequence if needed
        if isinstance(items, Block):
            items_seq = items.to_list()
        else:
            items_seq = items

        if len(items_seq) != self._span.length:
            raise ValueError("Length mismatch in store()")

        # Check write access first (provides better error message for NA state)
        self._check_can_write()

        # Determine if this is the first store(acc=True) by checking if we're in WO state
        is_first_acc_store = acc and self._access_state == AccessState.WO

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

        Broadcasting must be explicit via ttl.math.broadcast().
        Implicit broadcasting (different shapes without explicit broadcast) is an error.
        """
        len_left = len(left)
        len_right = len(right)
        left_shape = left._shape
        right_shape = right._shape

        # Check if shapes match exactly
        if left_shape == right_shape and len_left == len_right:
            return [op(left[i], right[i]) for i in range(len_left)]

        # Check if one operand is marked for broadcasting
        left_broadcast = getattr(left, "_broadcast_dims", None)
        right_broadcast = getattr(right, "_broadcast_dims", None)

        # Exactly one operand should be marked for broadcast
        if left_broadcast and right_broadcast:
            raise ValueError(
                f"Cannot perform operation: both operands are marked for broadcast "
                f"(left dims={left_broadcast}, right dims={right_broadcast}). "
                f"Only one operand should be marked for broadcast."
            )

        if not left_broadcast and not right_broadcast:
            raise ValueError(
                f"Cannot perform operation: shape mismatch {left_shape} vs {right_shape}. "
                f"Use ttl.math.broadcast() to explicitly broadcast one of the operands."
            )

        # One operand is marked for broadcast - handle symmetrically
        broadcast_block = left if left_broadcast else right
        other_block = right if left_broadcast else left
        broadcast_dims = left_broadcast or right_broadcast
        broadcast_shape = broadcast_block._shape
        other_shape = other_block._shape

        # Validate broadcast
        if len(broadcast_shape) != len(other_shape):
            raise ValueError(
                f"Cannot broadcast: dimension mismatch between shapes {broadcast_shape} and {other_shape}"
            )

        for dim in broadcast_dims:
            if dim >= len(broadcast_shape):
                raise ValueError(
                    f"Cannot broadcast: dimension {dim} out of range for shape {broadcast_shape}"
                )
            if broadcast_shape[dim] != 1:
                raise ValueError(
                    f"Cannot broadcast: dimension {dim} must have size 1, got {broadcast_shape[dim]}"
                )

        # Perform the operation with broadcasting
        from .ttnnsim import broadcast_tensors

        return broadcast_tensors(
            left.to_list(), right.to_list(), left_shape, right_shape, op
        )

    def _binary_op(
        self,
        other: "Block",
        op: Callable[[Any, Any], Any],
    ) -> "Block":
        """Element-wise binary op: self (op) other."""
        result_list = self._apply_binary_op(self, other, op)

        # Infer result shape using broadcasting rules
        result_shape = self._infer_broadcast_shape(self._shape, other._shape)

        return Block.from_list(result_list, result_shape)

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
            return Block.from_list(result_tensors, shape=self._shape)

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
