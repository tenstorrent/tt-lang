# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Block, ring-buffer primitives, and high-level DataflowBuffer interface.

This module provides:
- Block: a logically contiguous window into a ring buffer with state machine enforcement
- DFBStats: statistics snapshot for a dataflow buffer
- DataflowBuffer: high-level tensor-aware dataflow buffer wrapper
"""

import math
import operator as _op
from itertools import product as _product
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Union,
)

import torch

from pydantic import validate_call

from .blockstate import (
    AccessState,
    BlockAcquisition,
    ExpectedOp,
    ThreadType,
    STATE_TRANSITIONS,
    get_current_thread_type,
)
from .dfbstate import DFBState
from .constants import TILE_SHAPE
from .errors import DFBContractError
from .stats import record_dfb_reserve, record_dfb_wait
from .ttnnsim import Tensor, tile_count_from_tensor
from .typedefs import Index, Shape, Size


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
        "_shape",
        "_acquisition",
        "_thread_type",
        "_access_state",
        "_expected_ops",
        "_is_temporary",
        "_source_blocks",  # Track wait() blocks that contributed to this temporary block
        "dfb",  # Reference to DataflowBuffer for context manager cleanup
        "dfb_state",  # DFBState reference for updating ring-buffer slot on copy_as_dest
        "dfb_slot_idx",  # Index of this block's slot in the ring buffer
    )

    # TODO: We can't do @validate_call here. There reason is that @validate_call actually
    #       copies the arguments to validate them and returns the copies to the decorated
    #       function. In our case, we don't want the copy of the tensor, we want to use the
    #       original tensor as is. This is a limitation of pydantic's validate_call, and
    #       perhaps a good reason to look for other frameworks that don't do that! (beartype?)
    def __init__(
        self,
        tensor: Tensor,
        shape: Shape,
        acquisition: BlockAcquisition,
        thread_type: ThreadType,
        is_temporary: bool = False,
        dfb: Optional["DataflowBuffer"] = None,
    ):
        self._buf = tensor
        self._shape = shape
        self._is_temporary = is_temporary
        self._source_blocks: List["Block"] = []  # Track source wait() blocks
        self.dfb = dfb  # Reference to DataflowBuffer for context manager support
        self.dfb_state: Optional[DFBState] = None
        self.dfb_slot_idx: int = -1

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

        Only works for Blocks that came from DataflowBuffer wait()/reserve().
        Temporary blocks (from arithmetic operations) don't have cleanup actions.

        If an exception occurred in the with block, cleanup is skipped to preserve
        the exception and avoid state machine errors.
        """
        # Only perform cleanup if no exception occurred
        if exc_type is None and self.dfb is not None:
            # Block came from DFB - perform appropriate cleanup
            if self._acquisition == BlockAcquisition.RESERVE:
                self.push()
            elif self._acquisition == BlockAcquisition.WAIT:
                self.pop()

    def __repr__(self) -> str:
        acq = self._acquisition.name
        expected = {op.name for op in self._expected_ops}
        return (
            f"Block("
            f"shape={self._shape}, "
            f"data={repr(self._buf.to_torch())}, "
            f"acq={acq}, "
            f"thread={self._thread_type.name}, "
            f"access={self._access_state.name}, "
            f"expected={expected})"
        )

    def pop(self) -> None:
        if self.dfb is None:
            raise RuntimeError(
                "Block.pop() is only valid for blocks acquired from a DataflowBuffer."
            )
        self.dfb.pop_block()

    def push(self) -> None:
        if self.dfb is None:
            raise RuntimeError(
                "Block.push() is only valid for blocks acquired from a DataflowBuffer."
            )
        self.dfb.push_block()

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
        context_transitions = STATE_TRANSITIONS.get(context_key)

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
        return math.prod(self._shape)

    @property
    def is_temporary(self) -> bool:
        """Check if this Block is a temporary computation result (not DFB-backed)."""
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
        """Return a single tile by flat index (row-major order across self._shape).

        Delegates to to_list() which does not require tile-aligned dimensions.
        """
        self._check_can_read()
        n = math.prod(self._shape)
        if not (0 <= idx < n):
            raise IndexError(idx)
        return self.to_list()[idx]

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

    def to_list(self) -> List[Tensor]:
        """Split the backing Tensor into individual tiles.

        Returns tiles in row-major order across self._shape.  Each tile is a
        slice of the backing tensor — no tile-alignment constraints are imposed
        so non-standard tile sizes (e.g. in tests) are supported.
        """
        buf = self._buf.to_torch()
        shape = self._shape

        if len(shape) == 1:
            # 1-D: single tile-grid dimension, no batch dims.
            tk = shape[0]
            w = buf.shape[-1]
            tile_w = w // tk if tk > 0 else 1
            return [Tensor(buf[slice(c * tile_w, (c + 1) * tile_w)]) for c in range(tk)]

        nb = len(shape) - 2
        tm, tk = shape[nb], shape[nb + 1]
        h, w = buf.shape[-2], buf.shape[-1]
        tile_h = h // tm if tm > 0 else 1
        tile_w = w // tk if tk > 0 else 1

        tiles: List[Tensor] = []
        for coords in _product(*[range(d) for d in shape]):
            batch_idx = coords[:nb]
            r, c = coords[nb], coords[nb + 1]
            slices = (
                *batch_idx,
                slice(r * tile_h, (r + 1) * tile_h),
                slice(c * tile_w, (c + 1) * tile_w),
            )
            tiles.append(Tensor(buf[slices]))
        return tiles

    def to_tensor(self) -> Tensor:
        """Return the backing multi-tile Tensor directly."""
        return self._buf

    @classmethod
    def from_list(
        cls,
        tensors: List[Tensor],
        shape: Shape,
    ) -> "Block":
        """Create a temporary Block by assembling a list of tiles into a Tensor.

        Tiles must be in row-major order across shape.  The resulting Block
        owns a freshly assembled multi-tile Tensor with element shape derived
        from the individual tile sizes.
        """
        if len(shape) == 1:
            # 1-D: tiles are contiguous vectors; just concatenate along dim 0.
            elem_tensor = torch.cat([t.to_torch() for t in tensors], dim=0)
        else:
            nb = len(shape) - 2
            batch = shape[:nb]
            TM, TK = shape[nb], shape[nb + 1]
            first_raw = tensors[0].to_torch()
            tile_h, tile_w = first_raw.shape[-2], first_raw.shape[-1]
            stacked = torch.stack([t.to_torch() for t in tensors])
            tile_grid = stacked.reshape(*shape, tile_h, tile_w)
            # (*batch, TM, TK, r, c) -> (*batch, TM, r, TK, c) -> (*batch, TM*r, TK*c)
            perm = list(range(nb)) + [nb, nb + 2, nb + 1, nb + 3]
            elem_tensor = tile_grid.permute(*perm).reshape(
                *batch, TM * tile_h, TK * tile_w
            )
        return cls(
            tensor=Tensor(elem_tensor),
            shape=shape,
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.COMPUTE,
            is_temporary=True,
        )

    @classmethod
    def from_tensor(cls, t: Tensor) -> "Block":
        """Create a temporary Block wrapping a ttnnsim.Tensor.

        Infers the tile-grid shape from the tensor's element dimensions.
        The last two element dimensions must be multiples of TILE_SHAPE (or
        exactly 1 for degenerate tiles).

        Args:
            t: Source tensor.

        Returns:
            A temporary Block backed directly by t (no copy).

        Raises:
            ValueError: If the tensor dimensions are not tile-aligned.
        """
        elem_shape = t.shape
        if len(elem_shape) == 1:
            w = elem_shape[0]
            if w != 1 and w % TILE_SHAPE[0] != 0:
                raise ValueError(
                    f"1-D tensor dimension ({w},) must be a multiple of "
                    f"TILE_SHAPE[0]={TILE_SHAPE[0]}, or exactly 1"
                )
            tk = 1 if w == 1 else w // TILE_SHAPE[0]
            tile_shape: Shape = (tk,)
        else:
            h, w = elem_shape[-2], elem_shape[-1]
            if (h != 1 and h % TILE_SHAPE[0] != 0) or (
                w != 1 and w % TILE_SHAPE[1] != 0
            ):
                raise ValueError(
                    f"Last two tensor dimensions ({h}, {w}) must be multiples of "
                    f"TILE_SHAPE {TILE_SHAPE}, or exactly 1"
                )
            batch_shape = elem_shape[:-2]
            tm = 1 if h == 1 else h // TILE_SHAPE[0]
            tk = 1 if w == 1 else w // TILE_SHAPE[1]
            tile_shape = (*batch_shape, tm, tk)
        return cls(
            tensor=t,
            shape=tile_shape,
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.COMPUTE,
            is_temporary=True,
        )

    def copy_as_dest(self, tensor: Tensor) -> None:
        """Store tensor into this block as part of a copy operation.

        Used by copy handlers; does NOT update the state machine.  Validates
        that the source and destination represent the same number of tiles.

        If element shapes differ (e.g. degenerate vs. standard tiles), the
        backing tensor is replaced and the ring-buffer slot is updated so that
        wait() consumers see the correct tensor.

        Args:
            tensor: Tensor to store in this block.

        Raises:
            ValueError: If tensor tile count does not match this block's shape.
        """
        # Validate tile count compatibility: same number of tiles, same ndim
        src_tile_count = tile_count_from_tensor(tensor)
        dst_tile_count = math.prod(self._shape)
        if src_tile_count != dst_tile_count:
            raise ValueError(
                f"Shape mismatch in copy_as_dest(): "
                f"source tensor {tensor.shape} has {src_tile_count} tiles, "
                f"but block {self._shape} expects {dst_tile_count} tiles"
            )

        if tensor.shape == self._buf.shape:
            # Fast path: same element shape — copy data in-place
            self._buf.to_torch().copy_(tensor.to_torch())
        else:
            # Shape differs (e.g. degenerate tile): replace the tensor reference
            # and update the ring-buffer slot so consumers see the new tensor.
            self._buf = tensor
            if self.dfb_state is not None:
                self.dfb_state.buf[self.dfb_slot_idx] = tensor

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

    def store(self, items: "Block", acc: bool = False) -> None:
        """Store data into this block.

        Args:
            items: A Block whose tile count matches this block.
            acc: If True, accumulate with existing values (+=), otherwise
                 assign (=).  The first store(acc=True) always assigns.

        Raises:
            ValueError: If the source tile count does not match this block's.
        """
        # Check write access before touching items so state-machine errors are
        # always surfaced first.
        self._check_can_write()

        src_tensor = items._buf
        source_blocks_to_mark: List["Block"] = []
        # Track wait() Compute source blocks for state machine
        if (
            items._acquisition == BlockAcquisition.WAIT
            and items._thread_type == ThreadType.COMPUTE
            and ExpectedOp.STORE_SRC in items._expected_ops
        ):
            source_blocks_to_mark.append(items)
        elif items._is_temporary and items._source_blocks:
            source_blocks_to_mark.extend(
                blk
                for blk in items._source_blocks
                if ExpectedOp.STORE_SRC in blk._expected_ops
            )

        # Validate tile counts match
        src_tiles = tile_count_from_tensor(src_tensor)
        dst_tiles = math.prod(self._shape)
        if src_tiles != dst_tiles:
            raise ValueError(
                f"Shape mismatch in store(): "
                f"source {src_tensor.shape} ({src_tiles} tiles) vs "
                f"block {self._shape} ({dst_tiles} tiles)"
            )

        # Mark source wait() blocks as consumed
        for source_block in source_blocks_to_mark:
            source_block.mark_store_read_complete()

        is_first_acc_store = acc and self._access_state == AccessState.MW
        self.mark_store_complete(acc=acc)

        if acc and not is_first_acc_store:
            # Subsequent accumulate: y += x (requires matching shapes)
            self._buf.to_torch().add_(
                src_tensor.to_torch().reshape(self._buf.to_torch().shape)
            )
        elif src_tensor.shape == self._buf.shape:
            # Fast path: same element shape — copy in-place
            self._buf.to_torch().copy_(src_tensor.to_torch())
        else:
            # Degenerate tile: element shapes differ but tile counts match.
            # Replace the backing tensor and update the ring-buffer slot if needed.
            self._buf = src_tensor
            if self.dfb_state is not None:
                self.dfb_state.buf[self.dfb_slot_idx] = src_tensor

    def _binary_op(
        self,
        other: "Block",
        op: Callable[[Any, Any], Any],
    ) -> "Block":
        """Element-wise binary op: self (op) other.

        Applies op on the underlying Tensors (PyTorch broadcasting applies).
        Validates that tile-grid shapes are broadcast-compatible.

        Tracks wait() Compute blocks that contribute to the result.
        """
        left_shape = self._shape
        right_shape = other._shape

        # Validate broadcast compatibility at the tile-grid level
        if len(left_shape) != len(right_shape):
            raise ValueError(
                f"Cannot broadcast: dimension mismatch between shapes "
                f"{left_shape} and {right_shape}."
            )
        for i, (l_dim, r_dim) in enumerate(zip(left_shape, right_shape)):
            if l_dim != r_dim and l_dim != 1 and r_dim != 1:
                raise ValueError(
                    f"Cannot broadcast: incompatible shapes {left_shape} and "
                    f"{right_shape}. Dimension {i} has sizes {l_dim} and {r_dim}."
                )

        result_shape = self._infer_broadcast_shape(left_shape, right_shape)

        if left_shape == right_shape:
            # Fast path: shapes match, operate on packed tensors directly.
            result_block = Block(
                tensor=op(self._buf, other._buf),
                shape=result_shape,
                acquisition=BlockAcquisition.RESERVE,
                thread_type=ThreadType.COMPUTE,
                is_temporary=True,
            )
        else:
            # Tile-grid broadcasting: tile-grid dims are entangled with element
            # dims in the packed tensor, so PyTorch cannot broadcast them
            # automatically.  Fall back to tile-by-tile ops.
            from .ttnnsim import broadcast_tensors

            result_tiles = broadcast_tensors(
                self.to_list(), other.to_list(), left_shape, right_shape, op
            )
            result_block = Block.from_list(result_tiles, result_shape)

        # Track source wait() blocks that contributed to this result
        for block in [self, other]:
            if (
                not block._is_temporary
                and block._acquisition == BlockAcquisition.WAIT
                and block._thread_type == ThreadType.COMPUTE
            ):
                result_block._source_blocks.append(block)
            elif block._is_temporary:
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
                result_block = Block(
                    tensor=self._buf**other,
                    shape=self._shape,
                    acquisition=BlockAcquisition.RESERVE,
                    thread_type=ThreadType.COMPUTE,
                    is_temporary=True,
                )
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
                return self._binary_op(other, _op.pow)

    def __matmul__(self, other: "Block") -> "Block":
        # Matrix multiplication is not a broadcasting operation.
        # It has its own shape rules: (M, K) @ (K, N) -> (M, N).
        # matmul is defined later in this module (after Block and DataflowBuffer).
        return matmul(self, other)

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
    def raw_tensor(self) -> Tensor:
        """Return the backing multi-tile Tensor (for copy handlers and stats)."""
        return self._buf

    @property
    def expected_ops(self) -> set[ExpectedOp]:
        """Get the set of expected operations for this block."""
        return self._expected_ops

    @property
    def shape(self) -> Shape:
        """Get the shape (rows, cols in tiles) of this block from its associated DFB."""
        return self._shape


class DFBStats(NamedTuple):
    """Statistics for a dataflow buffer.

    All counts (capacity, visible, reserved, free) are in operations, where
    one operation equals tiles_per_op tiles (= math.prod(shape)).
    """

    capacity: int  # total slots (= buffer_factor)
    visible: int  # slots ready to consume
    reserved: int  # slots reserved for writing
    free: int  # slots available for reservation
    head: int  # current read slot index
    slots: List[
        Optional[Tensor]
    ]  # slot list: None=empty or a multi-tile Tensor (for debugging)


class DataflowBuffer:
    """
    Dataflow buffer for tensor-based producer/consumer data movement.

    Each DataflowBuffer owns its ring buffer state directly and manages a
    fixed-size ring buffer with space for a configurable number of tiles.
    Operations like wait() and reserve() work with a fixed number of tiles
    determined by the shape parameter.

    Example:
        dfb = DataflowBuffer(element=t, shape=(2, 3), buffer_factor=2)

        # Producer workflow
        write_view = dfb.reserve()  # Reserve space for 6 tiles
        # ... write data to write_view ...
        write_view.push()  # Make data visible

        # Consumer workflow
        read_view = dfb.wait()  # Wait for 6 tiles
        # ... read data from read_view ...
        read_view.pop()  # Free consumed tiles
    """

    def __init__(
        self,
        element: Tensor,
        shape: Shape,
        buffer_factor: Size = 2,
    ):
        """
        Initialize a DataflowBuffer.

        Args:
            element: Tensor used to determine dtype for zero-initialized reserved blocks
            shape: Tile-grid shape for each wait/reserve operation (at least 1 dimension)
            buffer_factor: Capacity multiplier (capacity = prod(shape) * buffer_factor)

        Raises:
            ValueError: If shape or buffer_factor are invalid
        """
        if len(shape) < 1:
            raise ValueError(f"Shape must have at least 1 dimension, got {shape}")
        if any(s <= 0 for s in shape):
            raise ValueError(f"Shape elements must be positive, got {shape}")
        if buffer_factor <= 0:
            raise ValueError(f"buffer_factor must be positive, got {buffer_factor}")

        self.element = element
        self._shape = shape
        self._buffer_factor = buffer_factor

        self._pending_reserved_block: Optional[Block] = None
        self._pending_waited_block: Optional[Block] = None

        # Create and configure the ring-buffer state immediately.
        self._state = DFBState()
        self._state.cap = buffer_factor
        self._state.shape = shape
        self._state.buf = [None] * buffer_factor
        self._state.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def wait(self) -> Block:
        """Wait for data to be available and return a read view.

        Blocks until one operation slot is visible, then returns a Block
        providing access to that slot's tiles.

        Usage:
            blk = dfb.wait()
            data = blk[0]
            blk.pop()  # manual pop required

        Returns:
            Block providing read access to the available tiles

        Raises:
            RuntimeError: If called again before pop()
        """
        if self._pending_waited_block is not None:
            raise RuntimeError(
                "Cannot call wait() again before pop(): "
                "DataflowBuffer already has a pending waited block. "
                "You must call pop() before calling wait() again."
            )

        from .greenlet_scheduler import block_if_needed

        block_if_needed(self, "wait")

        state = self._state
        assert state.visible >= 1, (
            f"wait: expected >=1 visible operations, got {state.visible}. "
            "block_if_needed() should have been called first."
        )
        slot = state.buf[state.head]
        assert slot is not None, "Visible slot has no data — internal inconsistency."
        thread_type = get_current_thread_type()
        block = Block(
            tensor=slot,
            shape=state.shape,
            acquisition=BlockAcquisition.WAIT,
            thread_type=thread_type,
        )
        block.dfb = self
        self._pending_waited_block = block

        record_dfb_wait(self, math.prod(state.shape))

        return block

    def can_wait(self) -> bool:
        """Check if wait() can proceed without blocking.

        Returns:
            True if at least one complete operation slot is ready to consume.
        """
        return self._state.visible >= 1

    def reserve(self) -> Block:
        """Reserve one operation slot for writing and return a write view.

        Blocks until a free slot is available. The slot is zero-initialized
        before being returned.

        Usage:
            blk = dfb.reserve()
            blk.store(data)
            blk.push()  # manual push required

        Returns:
            Block providing write access to the reserved slot

        Raises:
            RuntimeError: If called again before push()
        """
        if self._pending_reserved_block is not None:
            raise RuntimeError(
                "Cannot call reserve() again before push(): "
                "DataflowBuffer already has a pending reserved block. "
                "You must call push() before calling reserve() again."
            )

        from .greenlet_scheduler import block_if_needed

        block_if_needed(self, "reserve")

        state = self._state
        assert state.free() >= 1, (
            f"reserve: expected >=1 free operation slots, got {state.free()}. "
            "block_if_needed() should have been called first."
        )
        slot_idx = state.back_slot()
        # Build element-space shape.
        if len(state.shape) == 1:
            tk = state.shape[0]
            elem_shape = (tk * TILE_SHAPE[0],)
        else:
            nb = len(state.shape) - 2
            batch = state.shape[:nb]
            tm, tk = state.shape[nb], state.shape[nb + 1]
            elem_shape = (*batch, tm * TILE_SHAPE[0], tk * TILE_SHAPE[1])
        slot = Tensor(torch.zeros(elem_shape, dtype=self.element.dtype))
        state.buf[slot_idx] = slot
        state.reserved += 1

        thread_type = get_current_thread_type()
        block = Block(
            tensor=slot,
            shape=state.shape,
            acquisition=BlockAcquisition.RESERVE,
            thread_type=thread_type,
        )
        block.dfb = self
        block.dfb_state = state
        block.dfb_slot_idx = slot_idx

        self._pending_reserved_block = block

        record_dfb_reserve(self, math.prod(state.shape))

        return block

    def can_reserve(self) -> bool:
        """Check if reserve() can proceed without blocking.

        Returns:
            True if at least one operation slot is free.
        """
        return self._state.free() >= 1

    def push_block(self) -> None:
        """Make the reserved slot visible to consumers.

        Must be called after reserve() and writing data to the returned Block.

        Raises:
            DFBContractError: If no slot was reserved.
        """
        if self._pending_reserved_block is not None:
            self._pending_reserved_block.mark_push_complete()
            self._pending_reserved_block = None

        state = self._state
        if state.reserved < 1:
            raise DFBContractError("push_block: no reserved operation slot to push")
        state.reserved -= 1
        state.visible += 1

    def pop_block(self) -> None:
        """Free the consumed slot, advancing the read pointer.

        Must be called after wait() and reading data from the returned Block.

        Raises:
            DFBContractError: If no slot was waited on.
        """
        if self._pending_waited_block is not None:
            self._pending_waited_block.mark_pop_complete()
            self._pending_waited_block = None

        state = self._state
        if state.visible < 1:
            raise DFBContractError("pop_block: no visible operation slot to pop")
        state.buf[state.head] = None
        state.head = (state.head + 1) % state.cap
        state.visible -= 1

    @property
    def shape(self) -> Shape:
        """Get the shape (in tiles) for wait/reserve operations."""
        return self._shape

    @property
    def capacity_tiles(self) -> Size:
        """Get the total capacity of the buffer in tiles."""
        return self._state.cap * math.prod(self._state.shape)

    @property
    def buffer_factor(self) -> Size:
        """Get the buffer factor (capacity multiplier)."""
        return self._buffer_factor

    def stats(self) -> DFBStats:
        """Get current buffer statistics (all counts in operations)."""
        return DFBStats(
            capacity=self._state.cap,
            visible=self._state.visible,
            reserved=self._state.reserved,
            free=self._state.free(),
            head=self._state.head,
            slots=list(self._state.buf),
        )

    def reset(self) -> None:
        """Reset the dataflow buffer to its initial empty state."""
        self._state.reset()

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
                f"DataflowBuffer {self} has incomplete blocks at end of execution:\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "DataflowBuffer":
        """Return a fresh DataflowBuffer with the same configuration.

        Deep-copying a DataflowBuffer yields an independent buffer with the same
        shape/capacity settings and a clean ring-buffer state.
        """
        new_dfb = DataflowBuffer(
            element=self.element,
            shape=self._shape,
            buffer_factor=self._buffer_factor,
        )
        memo[id(self)] = new_dfb
        return new_dfb

    def __repr__(self) -> str:
        s = self._state
        return (
            f"DataflowBuffer("
            f"shape={self._shape}, "
            f"capacity={s.cap}, "
            f"available_for_wait={s.visible}, "
            f"reserved={s.reserved}, "
            f"available_for_reserve={s.free()}, "
            f"head={s.head})"
        )


def make_dataflow_buffer_like(
    element: Tensor,
    shape: Shape,
    buffer_factor: Size = 2,
) -> DataflowBuffer:
    """
    Create a DataflowBuffer with the same dtype as the element.

    Args:
        element: A tensor used to determine the DataflowBuffer's dtype
        shape: Tuple of tile-grid dimensions, e.g. (1,) for 1-D, (1, 1) for 2-D,
            (1, 1, 1) for 3-D, etc. The total buffer capacity is
            math.prod(shape) * buffer_factor blocks.
        buffer_factor: Multiplier for total buffer capacity

    Returns:
        A DataflowBuffer with dtype matching the element

    Example:
        x = ttnn.zeros((32, 32), dtype=ttnn.float32)
        x_dfb = make_dataflow_buffer_like(x, shape=(2, 2), buffer_factor=2)
    """
    return DataflowBuffer(element=element, shape=shape, buffer_factor=buffer_factor)


def track_source_blocks(result_block: Block, *input_blocks: Block) -> None:
    """Track source wait() blocks for proper state management.

    Adds input wait() blocks to the result block's _source_blocks list so that
    when the result is stored, the sources can be marked as consumed.

    Args:
        result_block: The result block to track sources for
        *input_blocks: Input blocks that contributed to the result
    """
    for block in input_blocks:
        is_temporary = getattr(block, "_is_temporary", None)
        if is_temporary is None:
            continue

        if (
            not is_temporary
            and getattr(block, "acquisition", None) == BlockAcquisition.WAIT
            and getattr(block, "thread_type", None) == ThreadType.COMPUTE
        ):
            source_blocks = getattr(result_block, "_source_blocks", None)
            if source_blocks is not None:
                source_blocks.append(block)
        elif is_temporary:
            actual_source = getattr(block, "_source_blocks", None)
            result_source = getattr(result_block, "_source_blocks", None)
            if actual_source is not None and result_source is not None:
                result_source.extend(actual_source)


def matmul(a: Block, b: Block, _output_hint: Optional[Block] = None) -> Block:
    """Matrix multiplication of two blocks.

    Converts each block to a ttnnsim.Tensor, delegates to torch.matmul via the
    @ operator, then converts the result back to a Block.  The output tile
    shape follows PyTorch's matmul broadcasting rules applied to the
    element-space tensors, with the last two dimensions divided by TILE_SHAPE.

    Args:
        a: First input block.
        b: Second input block.
        _output_hint: Optional output block hint (unused in simulator).

    Returns:
        Block whose tile shape corresponds to the matmul output shape.
    """
    result_block = Block.from_tensor(a.to_tensor() @ b.to_tensor())
    track_source_blocks(result_block, a, b)
    return result_block
