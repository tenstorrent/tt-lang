# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Block, circular buffer API, and high-level CircularBuffer interface.

This module provides:
- Block: a logically contiguous window into a ring buffer with state machine enforcement
- CBStats: statistics snapshot for a circular buffer
- CBAPI: low-level circular buffer simulator API
- CircularBuffer: high-level tensor-aware circular buffer wrapper
"""

import operator as _op
import threading
from typing import (
    Annotated,
    Any,
    Callable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch

from pydantic import Field, validate_call

from .blockstate import (
    AccessState,
    BlockAcquisition,
    ExpectedOp,
    ThreadType,
    _STATE_TRANSITIONS,
    clear_current_thread_type,
    get_current_thread_type,
    set_current_thread_type,
)
from .cbstate import CBSlot, CBState
from .constants import CB_DEFAULT_TIMEOUT, MAX_CBS, TILE_SHAPE
from .errors import CBContractError, CBTimeoutError
from .stats import record_cb_reserve, record_cb_wait
from .ttnnsim import Tensor
from .typedefs import CBID, Index, Shape, Size, Span


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
        "cb",  # Reference to CircularBuffer for context manager cleanup
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
        cb: Optional["CircularBuffer"] = None,
    ):
        self._buf = buf
        self._capacity = capacity
        self._span = span
        self._shape = shape
        self._is_temporary = is_temporary
        self._source_blocks: List["Block"] = []  # Track source wait() blocks
        self.cb = cb  # Reference to CircularBuffer for context manager cleanup

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
                # Can only be used as source in another block's store operation
                self._access_state = AccessState.MR
                self._expected_ops = {ExpectedOp.STORE_SRC}

    def __enter__(self) -> "Block":
        """Context manager entry - returns self for use in with statement."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Context manager exit - automatically calls push() or pop() based on acquisition type.

        Only works for Blocks that came from CircularBuffer wait()/reserve().
        Temporary blocks (from arithmetic operations) don't have cleanup actions.

        If an exception occurred in the with block, cleanup is skipped to preserve
        the exception and avoid state machine errors.
        """
        # Only perform cleanup if no exception occurred
        if exc_type is None and self.cb is not None:
            # Block came from CB - perform appropriate cleanup
            if self._acquisition == BlockAcquisition.RESERVE:
                self.push()
            elif self._acquisition == BlockAcquisition.WAIT:
                self.pop()

    def pop(self) -> None:
        if self.cb is None:
            raise RuntimeError(
                "Block.pop() is only valid for blocks acquired from a CircularBuffer."
            )
        self.cb.pop_block()

    def push(self) -> None:
        if self.cb is None:
            raise RuntimeError(
                "Block.push() is only valid for blocks acquired from a CircularBuffer."
            )
        self.cb.push_block()

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
            buf=cast(List[CBSlot], tensors),
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

    def _transition_state(
        self,
        operation_key: str,
        operation_display: str,
        expected_op: ExpectedOp,
    ) -> None:
        """Execute a state machine transition using the transition table.

        Args:
            operation_key: Key for looking up transition in the table (e.g., "copy_src", "store_dst", "store_dst_acc")
            operation_display: Human-readable operation name for error messages
            expected_op: Expected operation for validation

        Raises:
            RuntimeError: If the state transition is invalid
        """
        # Validate that this operation is expected
        self._validate_state(operation_display, expected_op)

        # Look up transitions for this acquisition/thread_type combination
        context_key = (self._acquisition, self._thread_type)
        context_transitions = _STATE_TRANSITIONS.get(context_key)

        if context_transitions is None:
            raise RuntimeError(
                f"Impossible! No transitions defined for: "
                f"Acquisition={self._acquisition.name}, "
                f"Thread={self._thread_type.name}"
            )

        # Look up specific transition
        transition_key = (operation_key, self._access_state)
        transition = context_transitions.get(transition_key)

        if transition is None:
            raise RuntimeError(
                f"Impossible! Invalid state for {operation_display}: "
                f"Acquisition={self._acquisition.name}, "
                f"Thread={self._thread_type.name}, "
                f"Access={self._access_state.name}"
            )

        # Execute transition
        new_access_state, new_expected_ops = transition
        self._access_state = new_access_state
        self._expected_ops = new_expected_ops

    def mark_copy_as_source(self) -> None:
        """Mark that this block is being used as a copy source."""
        self._transition_state("copy_src", "copy (as source)", ExpectedOp.COPY_SRC)

    def mark_copy_as_dest(self) -> None:
        """Mark that this block is being used as a copy destination."""
        self._transition_state("copy_dst", "copy (as destination)", ExpectedOp.COPY_DST)

    def mark_tx_wait_complete(self) -> None:
        """Mark that tx.wait() has completed for a copy operation."""
        self._transition_state("tx_wait", "tx.wait()", ExpectedOp.TX_WAIT)

    def mark_store_read_complete(self) -> None:
        """Mark that this block was used as source (input) in a store operation."""
        self._transition_state("store_src", "store (as source)", ExpectedOp.STORE_SRC)

    def mark_store_complete(self, acc: bool = False) -> None:
        """Mark that store() has completed on this block (as destination).

        Args:
            acc: Whether this is an accumulate store operation
        """
        if acc:
            operation_type = "store_dst_acc"
            operation_display = "store(acc=True)"
            expected_op = ExpectedOp.STORE_ACC
        else:
            operation_type = "store_dst"
            operation_display = "store(acc=False)"
            expected_op = ExpectedOp.STORE
        self._transition_state(operation_type, operation_display, expected_op)

    def mark_push_complete(self) -> None:
        """Mark that push() has completed.

        Valid states (per state machine diagram):
        - Must be a reserve() block (not wait())
        - Must have PUSH in expected operations
        """
        # Validate that operation is expected
        self._validate_state("push()", ExpectedOp.PUSH)

        # Additional validation: push() requires RESERVE acquisition
        if self._acquisition != BlockAcquisition.RESERVE:
            raise RuntimeError(
                f"Cannot perform push(): push() is only valid for reserve() blocks, "
                f"got {self._acquisition.name} block. "
                f"Current state: Thread={self._thread_type.name}, Access={self._access_state.name}"
            )

        # Transition to DONE (Out of Scope)
        self._access_state = AccessState.OS
        self._expected_ops = set()  # Empty = DONE

    def mark_pop_complete(self) -> None:
        """Mark that pop() has completed.

        Valid states (per state machine diagram):
        - Must be a wait() block (not reserve())
        - Must have POP in expected operations
        - Can pop from MR (never used as source), RW (used as source), or A (accumulated)
        """
        # Validate that operation is expected
        self._validate_state("pop()", ExpectedOp.POP)

        # Additional validation: pop() requires WAIT acquisition
        if self._acquisition != BlockAcquisition.WAIT:
            raise RuntimeError(
                f"Cannot perform pop(): pop() is only valid for wait() blocks, "
                f"got {self._acquisition.name} block. "
                f"Current state: Thread={self._thread_type.name}, Access={self._access_state.name}"
            )

        # Additional validation: Can only pop from MR, RW, or A states
        if self._access_state not in (AccessState.MR, AccessState.RW, AccessState.A):
            raise RuntimeError(
                f"Cannot perform pop(): Invalid access state {self._access_state.name}. "
                f"Expected MR (never used), RW (used as source), or A (accumulated)."
            )

        # Transition to DONE (Out of Scope)
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
        # Note: We allow writing in MR/RW/A states as appropriate for the operation

    def get_item(self, idx: Index) -> Tensor:
        """Internal method to get item with lock checking.

        This is used internally by copy handlers and other internal operations.
        External code should not index blocks.
        """
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

    @validate_call
    def __getitem__(self, idx: Index) -> Tensor:
        """Block indexing is not allowed. Blocks should be used as whole units."""
        raise RuntimeError(
            "Block indexing (block[index]) is not allowed. "
            "Blocks must be used as whole units in operations like store() or arithmetic. "
            "Use block directly without indexing."
        )

    # TODO: Why does validate_call fail here? Maybe because Tensor could
    # resolve to tensor which is similar to a list?
    # @validate_call
    def __setitem__(self, idx: Index, value: Tensor) -> None:
        """Direct assignment to Block is not allowed. Use store() or copy() instead."""
        raise RuntimeError(
            "Direct assignment to Block is not allowed. Use block.store() or copy() instead."
        )

    def write_slot(self, idx: Index, value: Tensor) -> None:
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
    def pop_idx(self, idx: Index) -> None:
        if not (0 <= idx < self._span.length):
            raise IndexError(idx)
        value = self._buf[(self._span.start + idx) % self._capacity]
        if value is None:
            raise ValueError(f"Popping uninitialized or consumed slot at index {idx}")
        self._buf[(self._span.start + idx) % self._capacity] = None

    def to_list(self) -> List[Tensor]:
        """Convert block contents to a list.

        This is a convenience method for tests and debugging.
        Returns the actual tensor values from the buffer.
        """
        return [self.get_item(i) for i in range(len(self))]

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

        # Check compatibility using pattern matching
        for l, r in zip(left_shape, right_shape):
            match (l, r):
                case (1, _) | (_, 1):
                    # One dimension is 1: broadcasting compatible
                    pass
                case (x, y) if x == y:
                    # Both dimensions equal: compatible
                    pass
                case _:
                    # Incompatible dimensions
                    raise ValueError(
                        f"Incompatible shapes for broadcasting: {left_shape} and {right_shape}"
                    )

        # Now construct result_shape knowing all dimensions are compatible
        result_shape: Shape = tuple(max(l, r) for l, r in zip(left_shape, right_shape))

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

        # Convert items to sequence
        items_seq: Sequence[Tensor]
        match items:
            case Block():
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
            case _:
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
                    self.write_slot(i, v)
            else:
                # Subsequent store(acc=True): Accumulate (y += x)
                for i, v in enumerate(items_seq):
                    self.write_slot(i, self.get_item(i) + v)
        else:
            # Regular assignment
            for i, v in enumerate(items_seq):
                self.write_slot(i, v)

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
            return [op(left.get_item(i), right.get_item(i)) for i in range(len_left)]

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
            if (
                not block._is_temporary
                and block._acquisition == BlockAcquisition.WAIT
                and block._thread_type == ThreadType.COMPUTE
            ):
                result_block._source_blocks.append(block)
            elif block._is_temporary:
                # Temporary blocks may have their own source blocks to propagate
                result_block._source_blocks.extend(block._source_blocks)

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
        match other:
            case int():
                # Scalar power - apply to each tensor in the block
                block_list = self.to_list()
                result_tensors: List[Tensor] = [t**other for t in block_list]
                result_block = Block.from_list(result_tensors, shape=self._shape)

                # Track source wait() blocks
                if (
                    not self._is_temporary
                    and self._acquisition == BlockAcquisition.WAIT
                    and self._thread_type == ThreadType.COMPUTE
                ):
                    result_block._source_blocks.append(self)
                elif self._is_temporary:
                    result_block._source_blocks.extend(self._source_blocks)

                return result_block
            case _:
                # Block power
                return self._binary_op(other, _op.pow)

    def __matmul__(self, other: "Block") -> "Block":
        # Matrix multiplication is not a broadcasting operation
        # It has its own shape rules: (M, K) @ (K, N) -> (M, N)
        # Delegate to the dedicated matmul implementation in math.py
        from . import math as sim_math

        return sim_math.matmul(self, other)

    @property
    def acquisition(self) -> BlockAcquisition:
        """Get the acquisition method (reserve or wait) of this block."""
        return self._acquisition

    @property
    def thread_type(self) -> ThreadType:
        """Get the thread type (DM or Compute) that acquired this block."""
        return self._thread_type

    @property
    def access_state(self) -> AccessState:
        """Get the current access state of this block."""
        return self._access_state

    @property
    def expected_ops(self) -> set[ExpectedOp]:
        """Get the set of expected operations for this block."""
        return self._expected_ops

    @property
    def shape(self) -> Shape:
        """Get the shape (rows, cols in tiles) of this block from its associated CB."""
        return self._shape


class CBStats(NamedTuple):
    """Statistics for a circular buffer."""

    capacity: int
    visible: int
    reserved: int
    free: int
    step: Optional[int]
    head: int
    list: List[Optional[object]]


class CBAPI:
    """Circular buffer simulator API interface with its own state pool.
    The simulator is based on the following API:
    https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/circular_buffers/circular_buffers.html

    CBAPI is not generic to allow heterogeneous CBState instances with different element types.
    Each CBState in the pool can have a different CBElemTypeVar parameter.
    """

    def __init__(self, timeout: Optional[float] = CB_DEFAULT_TIMEOUT):
        """Initialize simulator with optional per-instance timeout (seconds)."""

        self._pool: List[object] = [None] * MAX_CBS
        self._timeout: Optional[float] = timeout
        self._next_cb_id: CBID = 0
        self._cb_allocator_lock = threading.Lock()

    def allocate_cb_id(self) -> CBID:
        """Allocate a unique CB ID from this API instance. Thread-safe."""
        with self._cb_allocator_lock:
            cb_id = self._next_cb_id
            self._next_cb_id += 1
            if self._next_cb_id > MAX_CBS:
                raise RuntimeError(
                    f"Maximum number of circular buffers exceeded: {MAX_CBS}"
                )
            return cb_id

    @validate_call
    def host_configure_cb(
        self, cb_id: CBID, capacity_tiles: Size, shape: Shape
    ) -> None:
        # Lazily create CBState if not already created
        if self._pool[int(cb_id)] is None:
            self._pool[int(cb_id)] = CBState()
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.cap = capacity_tiles
            cb_state.shape = shape
            cb_state.reset()

    @validate_call
    def host_reset_cb(self, cb_id: CBID) -> None:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            if not cb_state.configured:
                raise CBContractError("CB not configured; cannot reset")
            cb_state.reset()

    @validate_call
    def cb_stats(self, cb_id: CBID) -> CBStats:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.require_configured()
            return CBStats(
                capacity=cb_state.cap,
                visible=cb_state.visible,
                reserved=cb_state.reserved,
                free=cb_state.free(),
                step=cb_state.step,
                head=cb_state.head,
                list=list(cb_state.buf),
            )

    @validate_call
    def cb_pages_available_at_front(self, cb_id: CBID, num_tiles: Size) -> bool:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            return cb_state.visible >= num_tiles

    @validate_call
    def cb_pages_reservable_at_back(self, cb_id: CBID, num_tiles: Size) -> bool:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            return cb_state.free() >= num_tiles

    @validate_call
    def cb_wait_front(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.can_consume:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            thread = threading.current_thread()
            if (cb_state.consumer_waiting is not None) and (
                cb_state.consumer_waiting != thread
            ):
                raise CBContractError(
                    "Only one consumer thread may wait on a DFB at a time"
                )
            cb_state.consumer_waiting = thread
            if cb_state.step is None:
                cb_state.step = num_tiles
            else:
                if num_tiles != cb_state.last_wait_target + cb_state.step:
                    raise CBContractError(
                        "cb_wait_front must be cumulative with an increment of the initial number of tiles"
                        " requested until a pop occurs"
                    )
            ok = cb_state.can_consume.wait_for(
                lambda: cb_state.visible >= num_tiles, timeout=self._timeout
            )
            if not ok:
                raise CBTimeoutError(f"cb_wait_front timed out after {self._timeout}s")
            cb_state.last_wait_target = num_tiles

    @validate_call
    def cb_reserve_back(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.can_produce:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            thread = threading.current_thread()
            if (cb_state.producer_reserving is not None) and (
                cb_state.producer_reserving != thread
            ):
                raise CBContractError(
                    "Only one producer thread may reserve on a DFB at a time"
                )
            cb_state.producer_reserving = thread
            if num_tiles < cb_state.reserved:
                raise CBContractError("reserve target cannot regress within epoch")
            ok = cb_state.can_produce.wait_for(
                lambda: cb_state.free() >= num_tiles, timeout=self._timeout
            )
            if not ok:
                raise CBTimeoutError(
                    f"cb_reserve_back timed out after {self._timeout}s"
                )
            cb_state.reserved = num_tiles
            cb_state.last_reserve_target = num_tiles

    @validate_call
    def cb_push_back(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            if num_tiles > cb_state.reserved:
                raise CBContractError(
                    f"cb_push_back({num_tiles}) exceeds reserved={cb_state.reserved}"
                )
            cb_state.reserved -= num_tiles
            cb_state.visible += num_tiles
            if cb_state.reserved == 0:
                cb_state.producer_reserving = None
            with cb_state.can_consume:
                cb_state.can_consume.notify_all()

    @validate_call
    def cb_pop_front(self, cb_id: CBID, num_tiles: Size) -> None:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.require_configured()
            cb_state.check_num_tiles(num_tiles)
            if num_tiles > cb_state.visible:
                raise CBContractError(
                    f"cb_pop_front({num_tiles}) exceeds visible={cb_state.visible}"
                )
            span = cb_state.front_span(num_tiles)
            thread_type = get_current_thread_type()
            view = Block(
                cb_state.buf,
                cb_state.cap,
                span,
                cb_state.shape,
                BlockAcquisition.WAIT,
                thread_type,
            )
            for i in range(len(view)):
                view.pop_idx(i)
            cb_state.head = (cb_state.head + num_tiles) % cb_state.cap
            cb_state.visible -= num_tiles
            cb_state.last_wait_target = 0
            if cb_state.visible == 0:
                cb_state.consumer_waiting = None
            with cb_state.can_produce:
                cb_state.can_produce.notify_all()

    @validate_call
    def get_read_ptr(self, cb_id: CBID) -> Block:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.require_configured()
            if cb_state.last_wait_target <= 0:
                raise CBContractError("get_read_ptr requires prior cb_wait_front")
            if cb_state.visible < cb_state.last_wait_target:
                raise CBContractError(
                    "read window invalidated; call cb_wait_front again"
                )
            span = cb_state.front_span(cb_state.last_wait_target)
            thread_type = get_current_thread_type()
            block = Block(
                cb_state.buf,
                cb_state.cap,
                span,
                cb_state.shape,
                BlockAcquisition.WAIT,
                thread_type,
            )
            return block

    @validate_call
    def get_write_ptr(self, cb_id: CBID) -> Block:
        cb_state: CBState = self._pool[int(cb_id)]  # type: ignore[assignment]
        with cb_state.lock:
            cb_state.require_configured()
            if cb_state.last_reserve_target <= 0:
                raise CBContractError("get_write_ptr requires prior cb_reserve_back")
            if cb_state.reserved < cb_state.last_reserve_target:
                raise CBContractError("write window invalidated; call cb_reserve again")
            span = cb_state.back_span(cb_state.last_reserve_target)
            thread_type = get_current_thread_type()
            block = Block(
                cb_state.buf,
                cb_state.cap,
                span,
                cb_state.shape,
                BlockAcquisition.RESERVE,
                thread_type,
            )
            return block

    @validate_call
    def set_timeout(self, seconds: Optional[Annotated[float, Field(gt=0)]]) -> None:
        """Set this simulator instance's timeout."""
        self._timeout = seconds

    def get_timeout(self) -> Optional[float]:
        """Return this simulator instance's timeout."""
        return self._timeout


# TODO: Should this class now be private?
class CircularBuffer:
    """
    High-level circular buffer interface for tensor operations.

    This class provides a convenient wrapper around the low-level CBAPI,
    handling CB allocation and providing tensor-aware operations.

    The CircularBuffer manages a fixed-size circular buffer with space for
    a configurable number of tiles. Operations like wait() and reserve()
    work with a fixed number of tiles determined by the shape parameter.

    Example:
        cb = CircularBuffer(shape=(2, 3), buffer_factor=2)

        # Producer workflow
        write_view = cb.reserve()  # Reserve space for 6 tiles
        # ... write data to write_view ...
        write_view.push()  # Make data visible

        # Consumer workflow
        read_view = cb.wait()  # Wait for 6 tiles
        # ... read data from read_view ...
        read_view.pop()  # Free consumed tiles
    """

    def __init__(
        self,
        element: Tensor,
        shape: Shape,
        buffer_factor: Size = 2,
        api: Optional[CBAPI] = None,
    ):
        """
        Initialize a CircularBuffer.

        Args:
            element: A tensor used to determine the dtype for zero-initialized tensors in reserved blocks
            shape: Tuple of (rows, cols) specifying the tile shape for wait/reserve operations
            buffer_factor: Multiplier for total buffer capacity (capacity = shape[0] * shape[1] * buffer_factor)
            api: Optional CBAPI instance to use. If None, uses the shared default instance.

        Raises:
            ValueError: If shape or buffer_factor are invalid
            RuntimeError: If CB allocation fails
        """
        if len(shape) != 2:
            raise ValueError(f"Shape must be a 2-tuple, got {shape}")

        self.element = element
        self._shape = shape
        self._buffer_factor = buffer_factor

        # Store API instance (may be None)
        self._api: Optional[CBAPI] = api

        # Track pending blocks for state machine completion
        # At most one pending reserved block and one pending waited block at a time
        self._pending_reserved_block: Optional[Block] = None
        self._pending_waited_block: Optional[Block] = None

        # Calculate total capacity in tiles
        self._tiles_per_operation = shape[0] * shape[1]
        self._capacity_tiles = self._tiles_per_operation * buffer_factor

        # Only allocate and configure if API is provided
        # If None, this will be done when the CB is copied by Program
        if self._api is not None:
            self._cb_id: Optional[CBID] = self._api.allocate_cb_id()
            self._api.host_configure_cb(self._cb_id, self._capacity_tiles, self._shape)
            # Reset the buffer to initialize with zero entries
            self._api.host_reset_cb(self._cb_id)
        else:
            self._cb_id: Optional[CBID] = None  # Placeholder until properly initialized

    def _ensure_initialized(self) -> Tuple[CBAPI, CBID]:
        """Verify that the CircularBuffer has been properly initialized with an API.

        Returns:
            Tuple of (api, cb_id) for use in operations

        Raises:
            RuntimeError: If the CB was not initialized with an API instance
        """
        if self._api is None or self._cb_id is None:
            raise RuntimeError(
                "CircularBuffer was not properly initialized with a CBAPI instance. "
                "This likely means it was created outside of a kernel context. "
                "CircularBuffers must be created within @ttl.kernel decorated functions."
            )
        return self._api, self._cb_id

    def wait(self) -> Block:
        """Wait for data to be available and return a read view.

        This method blocks until the required number of tiles (as specified by
        the shape parameter) are available for reading. It returns a Block
        that provides access to the available data.

        Usage:
            blk = cb.wait()
            data = blk[0]
            blk.pop()  # manual pop required

        Returns:
            Block providing read access to the available tiles

        Raises:
            CBTimeoutError: If the wait times out
            CBContractError: If called incorrectly (e.g., multiple concurrent waits)
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()

        # Enforce: at most one pending wait() operation at a time
        if self._pending_waited_block is not None:
            raise RuntimeError(
                "Cannot call wait() again before pop(): "
                "CircularBuffer already has a pending waited block. "
                "You must call pop() before calling wait() again."
            )

        # Block if data not available
        from .greenlet_scheduler import block_if_needed

        block_if_needed(self, "wait")

        api.cb_wait_front(cb_id, self._tiles_per_operation)
        block = api.get_read_ptr(cb_id)
        block.cb = self  # Set CB reference for context manager support
        self._pending_waited_block = block

        # Record wait statistics
        record_cb_wait(self, self._tiles_per_operation)

        return block

    def can_wait(self) -> bool:
        """
        Check if wait() can proceed without blocking.

        Returns:
            True if sufficient data is available for wait(), False otherwise

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        stats = api.cb_stats(cb_id)
        return stats.visible >= self._tiles_per_operation

    def reserve(self) -> Block:
        """
        Reserve space for writing and return a write view.

        This method blocks until there is sufficient space to write the required
        number of tiles (as specified by the shape parameter). It returns a Block
        that provides access to the reserved space.

        The reserved block is automatically initialized with zero tensors using
        TILE_SHAPE dimensions and the element's dtype before being returned.

        Usage:
            blk = cb.reserve()
            blk.store(data)
            blk.push()  # manual push required

        Returns:
            Block providing write access to the reserved space

        Raises:
            CBTimeoutError: If the reservation times out
            CBContractError: If called incorrectly (e.g., multiple concurrent reserves)
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()

        # Enforce: at most one pending reserve() operation at a time
        if self._pending_reserved_block is not None:
            raise RuntimeError(
                "Cannot call reserve() again before push(): "
                "CircularBuffer already has a pending reserved block. "
                "You must call push() before calling reserve() again."
            )

        # Block if space not available
        from .greenlet_scheduler import block_if_needed

        block_if_needed(self, "reserve")

        api.cb_reserve_back(cb_id, self._tiles_per_operation)
        block = api.get_write_ptr(cb_id)
        block.cb = self  # Set CB reference for context manager support

        # Initialize the reserved block with zero tensors
        zero_tensor = Tensor(torch.zeros(TILE_SHAPE, dtype=self.element.dtype))
        for i in range(len(block)):
            block.write_slot(i, zero_tensor)

        self._pending_reserved_block = block

        # Record reserve statistics
        record_cb_reserve(self, self._tiles_per_operation)

        return block

    def can_reserve(self) -> bool:
        """
        Check if reserve() can proceed without blocking.

        Returns:
            True if sufficient space is available for reserve(), False otherwise

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        stats = api.cb_stats(cb_id)
        return stats.free >= self._tiles_per_operation

    def push_block(self) -> None:
        """
        Finalize a write operation, making reserved data visible to consumers.

        This method must be called after reserve() and writing data to the
        returned Block. It advances the CB pointers and makes the written
        data available for consumers to read via wait().

        Raises:
            CBContractError: If called without a prior reserve() or if the
                           push amount exceeds what was reserved
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        # Update state machine for the pending reserved block
        if self._pending_reserved_block is not None:
            self._pending_reserved_block.mark_push_complete()
            self._pending_reserved_block = None

        api, cb_id = self._ensure_initialized()
        api.cb_push_back(cb_id, self._tiles_per_operation)

    def pop_block(self) -> None:
        """
        Finalize a read operation, freeing consumed data.

        This method must be called after wait() and reading data from the
        returned Block. It advances the CB pointers and frees the consumed
        tiles, making space available for producers.

        After calling pop(), the Block returned by the corresponding wait()
        points to stale data and should not be accessed.

        Raises:
            CBContractError: If called without a prior wait() or if the
                           pop amount exceeds what is visible
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        # Update state machine for the pending waited block
        if self._pending_waited_block is not None:
            self._pending_waited_block.mark_pop_complete()
            self._pending_waited_block = None

        api, cb_id = self._ensure_initialized()
        api.cb_pop_front(cb_id, self._tiles_per_operation)

    @property
    def shape(self) -> Tuple[Size, Size]:
        """Get the shape (in tiles) for wait/reserve operations."""
        return self._shape

    @property
    def capacity_tiles(self) -> Size:
        """Get the total capacity of the buffer in tiles."""
        return self._capacity_tiles

    @property
    def buffer_factor(self) -> Size:
        """Get the buffer factor (capacity multiplier)."""
        return self._buffer_factor

    @property
    def cb_id(self) -> Optional[CBID]:
        """Get the internal CB ID (for debugging/advanced use)."""
        return self._cb_id

    def stats(self):
        """Get current buffer statistics.

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        return api.cb_stats(cb_id)

    def reset(self) -> None:
        """Reset the circular buffer to initial state.

        Raises:
            RuntimeError: If CircularBuffer was not properly initialized with an API
        """
        api, cb_id = self._ensure_initialized()
        api.host_reset_cb(cb_id)

    def validate_no_pending_blocks(self) -> None:
        """Validate that there are no pending blocks.

        This should be called at the end of kernel execution to ensure
        all blocks have been properly completed through push() or pop().

        Raises:
            RuntimeError: If there are any pending blocks
        """
        errors: List[str] = []

        if self._pending_reserved_block is not None:
            block = self._pending_reserved_block
            errors.append(
                f"Pending reserved block: Block(acquisition={block.acquisition.name}, "
                f"thread={block.thread_type.name}, access={block.access_state.name}, "
                f"expected_ops={[op.name for op in block.expected_ops]}). "
                f"Did you forget to call push()?"
            )

        if self._pending_waited_block is not None:
            block = self._pending_waited_block
            errors.append(
                f"Pending waited block: Block(acquisition={block.acquisition.name}, "
                f"thread={block.thread_type.name}, access={block.access_state.name}, "
                f"expected_ops={[op.name for op in block.expected_ops]}). "
                f"Did you forget to call pop()?"
            )

        if errors:
            raise RuntimeError(
                f"CircularBuffer {self} has incomplete blocks at end of execution:\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

    def __repr__(self) -> str:
        return (
            f"CircularBuffer(cb_id={self._cb_id}, shape={self._shape}, "
            f"capacity_tiles={self._capacity_tiles}, buffer_factor={self._buffer_factor})"
        )


def make_circular_buffer_like(
    element: Tensor,
    shape: Shape,
    buffer_factor: Size = 2,
) -> CircularBuffer:
    """
    Create a CircularBuffer with the same dtype as the element.

    Args:
        element: A tensor used to determine the CircularBuffer's dtype
        shape: Tuple of (rows, cols) specifying the tile shape for wait/reserve operations
        buffer_factor: Multiplier for total buffer capacity (capacity = shape[0] * shape[1] * buffer_factor)

    Returns:
        A CircularBuffer with dtype matching the element

    Example:
        x = ttnn.zeros((32, 32), dtype=ttnn.float32)
        x_cb = make_circular_buffer_like(x, shape=(2, 2), buffer_factor=2)
    """
    return CircularBuffer(element=element, shape=shape, buffer_factor=buffer_factor)
