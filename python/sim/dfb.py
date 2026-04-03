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
    cast,
)

import torch

from pydantic import validate_call

from .blockstate import (
    AccessState,
    BlockAcquisition,
    BlockStateMachine,
    ExpectedOp,
    ThreadType,
)
from .context import get_current_thread_type
from .dfbstate import DFBState
from .constants import TILE_SHAPE
from .errors import DFBContractError
from .stats import record_dfb_reserve, record_dfb_wait
from .ttnnsim import (
    ROW_MAJOR_LAYOUT,
    TILE_LAYOUT,
    Tensor,
    tile_count_from_tensor,
    tile_shape_from_tensor,
)
from .typedefs import Index, IndexType, PositiveInt, Shape, Size


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
        "_element_shape",  # Element-level shape (for broadcast semantics)
        "_sm",  # BlockStateMachine: owns all access-state logic
        "_is_temporary",
        "_store_confirmation_pending",  # Set by assign_src; cleared by mark_store_read_complete
        "_source_blocks",  # Track wait() blocks that contributed to this temporary block
        "_broadcast_dims",  # Pending broadcast dimensions (None or tuple of ints)
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
        # Element shape is always derived from the tensor's actual shape
        self._element_shape = tuple(tensor.shape)
        self._is_temporary = is_temporary
        self._store_confirmation_pending: bool = (
            False  # Set by assign_src; cleared by mark_store_read_complete
        )
        self._source_blocks: List["Block"] = []  # Track source wait() blocks
        self._broadcast_dims: Optional[tuple[int, ...]] = (
            None  # Pending broadcast metadata
        )
        self.dfb = dfb  # Reference to DataflowBuffer for context manager support
        self.dfb_state: Optional[DFBState] = None
        self.dfb_slot_idx: int = -1

        # Delegate all access-state logic to BlockStateMachine
        self._sm: BlockStateMachine = BlockStateMachine(acquisition, thread_type)
        if not is_temporary:
            self._sm.initialize()
        else:
            self._sm.set_unrestricted()

    # ------------------------------------------------------------------
    # Property proxies onto the state machine (used by dfb.py internals,
    # tests, and the public API properties further below).
    # ------------------------------------------------------------------

    @property
    def _acquisition(self) -> BlockAcquisition:
        return self._sm.acquisition

    @property
    def _thread_type(self) -> ThreadType:
        return self._sm.thread_type

    @property
    def _access_state(self) -> AccessState:
        return self._sm.access_state

    @property
    def _expected_ops(self) -> set[ExpectedOp]:
        return self._sm.expected_ops

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

    def mark_copy_as_source(self) -> None:
        """Mark that this block is being used as a copy source."""
        self._sm.transition("copy_src", "copy (as source)", ExpectedOp.COPY_SRC)

    def mark_copy_as_dest(self) -> None:
        """Mark that this block is being used as a copy destination."""
        self._sm.transition("copy_dst", "copy (as destination)", ExpectedOp.COPY_DST)

    def mark_tx_wait_complete(self) -> None:
        """Mark that tx.wait() has completed for a copy operation."""
        self._sm.transition("tx_wait", "tx.wait()", ExpectedOp.TX_WAIT)

    def mark_assign_src_complete(self) -> None:
        """Mark that this block's data was used as an arithmetic operand.

        Fires the assign_src state machine transition, unlocking pop() so the
        context manager can exit.  Sets _store_confirmation_pending and
        registers the block in the DFB's pending confirmation set.  Both are
        cleared by mark_store_read_complete() when store() eventually fires on
        the downstream result.  Program termination validates that all pending
        confirmations have been cleared.
        """
        self._sm.transition_assign_src()
        self._store_confirmation_pending = True
        if self.dfb is not None:
            self.dfb.register_pending_confirmation(self)

    def mark_store_read_complete(self) -> None:
        """Mark that this block was used as source (input) in a store operation.

        Fires the store_src state machine transition when the block is still
        active (not OS).  Always clears _store_confirmation_pending and removes
        the block from the DFB's pending confirmation set, satisfying the
        program-termination check regardless of whether the block has already
        been popped.
        """
        if self._sm.access_state != AccessState.OS:
            self._sm.transition("store_src", "store (as source)", ExpectedOp.STORE_SRC)
        self._store_confirmation_pending = False
        if self.dfb is not None:
            self.dfb.discard_pending_confirmation(self)

    def mark_store_complete(self) -> None:
        """Mark that store() has completed on this block (as destination)."""
        self._sm.transition("store_dst", "store()", ExpectedOp.STORE)

    def mark_push_complete(self) -> None:
        """Mark that push() has completed (RESERVE blocks only)."""
        self._sm.transition_push()

    def mark_pop_complete(self) -> None:
        """Mark that pop() has completed (WAIT blocks only)."""
        self._sm.transition_pop()

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
        # ROR state (async read in progress) allows reads since we're copying FROM this block

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
        if self._access_state in (AccessState.ROR, AccessState.OS):
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
        # Note: We allow writing in MR/RW states as appropriate for the operation

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
        """Split the backing Tensor into logical units.

        For TILE_LAYOUT blocks: returns one Tensor per tile in row-major order
        across self._shape.  No tile-alignment constraints are imposed so
        non-standard tile sizes (e.g. in tests) are supported.

        For ROW_MAJOR_LAYOUT blocks: returns one Tensor per row, where a row
        is a slice along the last dimension.  A 1-D block of shape (N,) returns
        one tensor of shape (N,); a block of shape (A, B, N) returns A*B
        tensors each of shape (N,).
        """
        buf = self._buf.to_torch()
        shape = self._shape

        if self.layout == ROW_MAJOR_LAYOUT:
            if len(shape) == 1:
                # 1-D: the entire buffer is a single row.
                return [Tensor(buf, ROW_MAJOR_LAYOUT)]
            # ND: iterate over all leading dimensions, yield one row per combination.
            rows: List[Tensor] = []
            for coords in _product(*[range(d) for d in shape[:-1]]):
                rows.append(Tensor(buf[coords], ROW_MAJOR_LAYOUT))
            return rows

        # TILE_LAYOUT path
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
        """Create a temporary Block by assembling a list of logical units.

        For TILE_LAYOUT tensors: tiles must be in row-major order across shape.
        The resulting Block owns a freshly assembled multi-tile Tensor with
        element shape derived from the individual tile sizes.

        For ROW_MAJOR_LAYOUT tensors: rows must be in row-major order across
        shape[:-1].  Each tensor must have shape (shape[-1],).  A 1-D shape
        (N,) expects a single tensor of shape (N,); an ND shape (A, B, N)
        expects A*B tensors each of shape (N,).
        """
        layout = tensors[0].layout if tensors else TILE_LAYOUT

        if layout == ROW_MAJOR_LAYOUT:
            if len(shape) == 1:
                # 1-D: single row, use it directly.
                elem_tensor = tensors[0].to_torch()
            else:
                # ND: stack rows and reshape to (*shape).
                elem_tensor = torch.stack([t.to_torch() for t in tensors]).reshape(
                    *shape
                )
            block = cls(
                tensor=Tensor(elem_tensor, ROW_MAJOR_LAYOUT),
                shape=shape,
                acquisition=BlockAcquisition.RESERVE,
                thread_type=ThreadType.COMPUTE,
                is_temporary=True,
            )
            return block

        # TILE_LAYOUT path
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

        # Create block with derived element shape
        block = cls(
            tensor=Tensor(elem_tensor),
            shape=shape,
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.COMPUTE,
            is_temporary=True,
        )
        return block

    @classmethod
    def from_tensor(cls, t: Tensor) -> "Block":
        """Create a temporary Block wrapping a ttnnsim.Tensor.

        For TILE_LAYOUT tensors the tile-grid shape is inferred from the
        element dimensions (last two must be multiples of TILE_SHAPE or
        exactly 1 for degenerate tiles).

        For ROW_MAJOR_LAYOUT tensors the tensor shape is used directly as
        the block shape (each dimension is already in scalar units).

        Args:
            t: Source tensor.

        Returns:
            A temporary Block backed directly by t (no copy).

        Raises:
            ValueError: If a TILE_LAYOUT tensor's dimensions are not tile-aligned.
        """
        if t.layout == ROW_MAJOR_LAYOUT:
            return cls(
                tensor=t,
                shape=t.shape,
                acquisition=BlockAcquisition.RESERVE,
                thread_type=ThreadType.COMPUTE,
                is_temporary=True,
            )

        elem_shape = t.shape
        if len(elem_shape) == 1:
            w = elem_shape[0]
            if w != 1 and w % TILE_SHAPE[0] != 0:
                raise ValueError(
                    f"1-D tensor dimension ({w},) must be a multiple of "
                    f"TILE_SHAPE[0]={TILE_SHAPE[0]}, or exactly 1"
                )
        else:
            h, w = elem_shape[-2], elem_shape[-1]
            if (h != 1 and h % TILE_SHAPE[0] != 0) or (
                w != 1 and w % TILE_SHAPE[1] != 0
            ):
                raise ValueError(
                    f"Last two tensor dimensions ({h}, {w}) must be multiples of "
                    f"TILE_SHAPE {TILE_SHAPE}, or exactly 1"
                )
        tile_shape: Shape = tile_shape_from_tensor(t)

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
        # Layouts must match - cannot copy between tiled and row-major endpoints.
        if tensor.layout != self.layout:
            raise ValueError(
                f"Layout mismatch in copy_as_dest(): "
                f"source tensor has layout {tensor.layout.name}, "
                f"but block has layout {self.layout.name}"
            )

        from .ttnnsim import check_count_match

        check_count_match(
            tile_count_from_tensor(tensor),
            math.prod(self._shape),
            self.layout,
            f"source tensor {tensor.shape}",
            f"block {self._shape}",
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

    @staticmethod
    def _expand_broadcast_dims(
        block: "Block",
        target_shape: Shape,
        target_element_shape: Shape,
        broadcast_dims: tuple[int, ...],
    ) -> Tensor:
        """Expand a block along broadcast dimensions to match target shape.

        Uses PyTorch broadcasting to expand the block's tensor from its current
        element shape to the target element shape. All validation is performed
        by broadcast() before setting _broadcast_dims metadata.

        Uses innermost-first convention: dims=[0] = last dimension in shape.

        Args:
            block: Source block with broadcast metadata
            target_shape: Target tile shape to expand to
            target_element_shape: Target element shape to expand to
            broadcast_dims: Dimensions to expand (innermost-first indexing)

        Returns:
            Tensor with expanded element shape
        """

        # Use PyTorch broadcasting to expand the tensor
        src_tensor = block._buf.to_torch()
        expanded_tensor = src_tensor.expand(*target_element_shape)

        # Return tensor directly
        return Tensor(expanded_tensor.contiguous())

    def store(self, items: "Block") -> None:
        """Store data into this block.

        Args:
            items: A Block whose tile count matches this block.

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
                or blk._store_confirmation_pending
            )

        # Check if source has broadcast metadata - if so, expand it
        src_shape = items._shape
        dst_shape = self._shape

        if hasattr(items, "_broadcast_dims") and items._broadcast_dims is not None:
            # Source came from broadcast() - expand using metadata
            src_tensor = self._expand_broadcast_dims(
                items, dst_shape, self._element_shape, items._broadcast_dims
            )
        else:
            # No broadcast metadata - validate tile counts match (allows different dimensionality)
            src_tiles = math.prod(src_shape)
            dst_tiles = math.prod(dst_shape)
            if src_tiles != dst_tiles:
                raise ValueError(
                    f"Shape mismatch in store(): "
                    f"source shape {src_shape} ({src_tiles} tiles) does not match "
                    f"destination shape {dst_shape} ({dst_tiles} tiles). "
                    f"Use broadcast() to expand the source if needed."
                )

        # Mark source wait() blocks as consumed
        for source_block in source_blocks_to_mark:
            source_block.mark_store_read_complete()

        self.mark_store_complete()

        if src_tensor.shape == self._buf.shape:
            # Fast path: same element shape — copy in-place
            self._buf.to_torch().copy_(src_tensor.to_torch())
        else:
            # Degenerate tile: element shapes differ but tile counts match.
            # Replace the backing tensor and update the ring-buffer slot if needed.
            self._buf = src_tensor
            if self.dfb_state is not None:
                self.dfb_state.buf[self.dfb_slot_idx] = src_tensor

    @staticmethod
    def _track_sources_for_result(result_block: "Block", *sources: "Block") -> None:
        """Track source blocks that contribute to a result block.

        For each source block, if it's a wait() Compute block, add it to the result's
        source list and eagerly mark it as read so that pop() can succeed when the
        block's context manager exits (i.e. before store() is called on the result).
        If it's already a temporary block, extend with its sources.

        Args:
            result_block: The result block to track sources for
            sources: Source blocks that contribute to the result
        """
        for source in sources:
            if (
                not source._is_temporary
                and source._acquisition == BlockAcquisition.WAIT
                and source._thread_type == ThreadType.COMPUTE
            ):
                result_block._source_blocks.append(source)
                # Fire assign_src so pop() is allowed when the 'with' context exits,
                # even though store() on the result block may come later.
                # The block is registered as pending store confirmation and cleared
                # by mark_store_read_complete() when store() eventually fires.
                if ExpectedOp.STORE_SRC in source._expected_ops:
                    source.mark_assign_src_complete()
            elif source._is_temporary:
                result_block._source_blocks.extend(source._source_blocks)

    def _create_temporary_result(
        self, result_tensor: Tensor, shape: Shape, *additional_sources: "Block"
    ) -> "Block":
        """Create a temporary result block with proper source tracking.

        Args:
            result_tensor: The computed result tensor
            shape: The shape of the result block
            additional_sources: Additional source blocks (for binary/ternary ops)

        Returns:
            A temporary Block with source blocks tracked
        """
        result_block = Block(
            tensor=result_tensor,
            shape=shape,
            acquisition=BlockAcquisition.RESERVE,
            thread_type=ThreadType.COMPUTE,
            is_temporary=True,
        )
        # Track all source blocks (self + any additional)
        self._track_sources_for_result(result_block, self, *additional_sources)
        return result_block

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

        # Check if either operand has broadcast metadata
        left_has_broadcast = (
            hasattr(self, "_broadcast_dims") and self._broadcast_dims is not None
        )
        right_has_broadcast = (
            hasattr(other, "_broadcast_dims") and other._broadcast_dims is not None
        )

        if left_has_broadcast and right_has_broadcast:
            raise ValueError(
                f"Cannot perform binary operation: both operands have pending broadcast. "
                f"Materialize one operand first by storing it."
            )

        # Expand operand with broadcast metadata to match the other
        left_buf = self._buf
        right_buf = other._buf
        result_shape = left_shape

        if left_has_broadcast:
            # Expand left to match right shape
            assert self._broadcast_dims is not None  # Checked by left_has_broadcast
            left_buf = self._expand_broadcast_dims(
                self, right_shape, other._element_shape, self._broadcast_dims
            )
            result_shape = right_shape
        elif right_has_broadcast:
            # Expand right to match left shape
            assert other._broadcast_dims is not None  # Checked by right_has_broadcast
            right_buf = self._expand_broadcast_dims(
                other, left_shape, self._element_shape, other._broadcast_dims
            )
            result_shape = left_shape
        elif left_shape != right_shape:
            # No broadcast metadata and shapes don't match - error
            raise ValueError(
                f"Shape mismatch in binary operation: left shape {left_shape} does not match "
                f"right shape {right_shape}. Use broadcast() to expand operands."
            )

        # Perform operation
        return self._create_temporary_result(
            op(left_buf, right_buf), result_shape, other
        )

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

    def __pow__(self, other: Union["Block", "PositiveInt"]) -> "Block":
        """Element-wise exponentiation.

        Supports both Block and scalar positive integer exponents.
        """
        match other:
            case int():
                return self._create_temporary_result(self._buf**other, self._shape)
            case _:
                return self._binary_op(other, _op.pow)

    def __neg__(self) -> "Block":
        """Unary negation (-block)."""
        return self._create_temporary_result(-self._buf, self._shape)

    def __abs__(self) -> "Block":
        """Absolute value (abs(block))."""
        return self._create_temporary_result(abs(self._buf), self._shape)

    def __matmul__(self, other: "Block") -> "Block":
        # Matrix multiplication is not a broadcasting operation.
        # It has its own shape rules: (M, K) @ (K, N) -> (M, N).
        # matmul is defined later in this module (after Block and DataflowBuffer).
        return matmul(self, other)

    # ---- in-place operator (temporary blocks only) ----

    def __iadd__(self, other: "Block") -> "Block":
        """In-place add for temporary accumulator blocks.

        Allows the pattern:
            y = ttl.math.fill(0)
            y += a_blk @ b_blk   # repeated accumulation
            dst_blk.store(y)

        Raises:
            RuntimeError: If self is not a temporary block.
        """
        if not self._is_temporary:
            raise RuntimeError(
                "In-place accumulation (+=) is only valid for temporary blocks "
                "(e.g. the result of fill() or a block expression). "
                "Use store() to write into a reserved block."
            )
        return self + other

    # ---- reflected operators (scalar op Block, e.g. fill(0) + blk) ----

    def __radd__(self, other: "Union[int, float]") -> "Block":
        """Scalar + Block: fill(v) + blk creates a temporary block with v added."""
        result_tensor = Tensor(
            torch.tensor(other, dtype=self._buf.dtype) + self._buf.to_torch()
        )
        return self._create_temporary_result(result_tensor, self._shape)

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

    @property
    def layout(self) -> IndexType:
        """Get the layout of the backing tensor (TILE_LAYOUT or ROW_MAJOR_LAYOUT)."""
        return self._buf.layout


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
        dfb = DataflowBuffer(likeness_tensor=t, shape=(2, 3), buffer_factor=2)

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
        likeness_tensor: Tensor,
        shape: Shape,
        buffer_factor: Size = 2,
    ):
        """
        Initialize a DataflowBuffer.

        Args:
            likeness_tensor: Tensor providing dtype and element shape (including degenerate dimensions)
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

        self.likeness_tensor = likeness_tensor
        self._shape = shape
        self._buffer_factor = buffer_factor

        if likeness_tensor.layout == ROW_MAJOR_LAYOUT:
            # Row-major: shape is in scalar units. No tile alignment required.
            # The likeness tensor supplies only dtype; its rank may differ from
            # shape (e.g. a full (N, H, W, C) tensor used as likeness for a
            # per-pixel block of shape (C,)).
            self._element_shape = shape
        else:
            # Tiled: validate tile alignment and derive element shape.
            if len(likeness_tensor.shape) != len(shape):
                raise ValueError(
                    f"Element shape dimensionality {len(likeness_tensor.shape)} does not match "
                    f"tile shape dimensionality {len(shape)}. Element shape: {likeness_tensor.shape}, "
                    f"tile shape: {shape}"
                )

            TILE_SIZE = TILE_SHAPE[0]  # 32
            ndims = len(shape)
            for i, (edim, tdim) in enumerate(zip(likeness_tensor.shape, shape)):
                if edim == 1:
                    # Degenerate dimension: tile dimension must also be 1
                    if tdim != 1:
                        raise ValueError(
                            f"Element shape dimension {i} is degenerate (size 1), but tile dimension is {tdim} (expected 1). "
                            f"Element shape: {likeness_tensor.shape}, tile shape: {shape}"
                        )
                elif i == ndims - 1 or i == ndims - 2:
                    # Last two dimensions are tile dimensions
                    if edim % TILE_SIZE != 0:
                        raise ValueError(
                            f"Element shape dimension {i} has size {edim}, which is not a multiple of TILE_SIZE ({TILE_SIZE}). "
                            f"Element shape: {likeness_tensor.shape}, tile shape: {shape}"
                        )
                    if edim // TILE_SIZE < tdim:
                        raise ValueError(
                            f"Element shape dimension {i} has {edim // TILE_SIZE} tiles, but tile shape requires at least {tdim} tiles. "
                            f"Element shape: {likeness_tensor.shape}, tile shape: {shape}"
                        )
                else:
                    # Batch/other dimensions
                    if edim < tdim:
                        raise ValueError(
                            f"Element shape dimension {i} has size {edim}, but tile shape requires at least {tdim}. "
                            f"Element shape: {likeness_tensor.shape}, tile shape: {shape}"
                        )

            self._element_shape = tuple(
                1 if edim == 1 else tdim * TILE_SIZE
                for edim, tdim in zip(likeness_tensor.shape, shape)
            )

        self._pending_reserved_block: Optional[Block] = None
        self._pending_waited_block: Optional[Block] = None
        self._pending_confirmations: set[Block] = set()

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

    def register_pending_confirmation(self, block: Block) -> None:
        """Register block as pending store confirmation."""
        self._pending_confirmations.add(block)

    def discard_pending_confirmation(self, block: Block) -> None:
        """Remove block from pending store confirmation set."""
        self._pending_confirmations.discard(block)

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
        # Create tensor with the DFB's element shape and layout
        slot = Tensor(
            torch.zeros(self._element_shape, dtype=self.likeness_tensor.dtype),
            self.likeness_tensor.layout,
        )
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
    def capacity_bytes(self) -> int:
        """Get the total L1 memory used by this buffer in bytes.

        Computed as: buffer_factor * elements_per_operation * bytes_per_element,
        where elements_per_operation is the product of the element shape dimensions.
        """
        return (
            self._buffer_factor
            * math.prod(self._element_shape)
            * self.likeness_tensor.element_size
        )

    @property
    def buffer_factor(self) -> Size:
        """Get the buffer factor (capacity multiplier)."""
        return self._buffer_factor

    @property
    def head(self) -> int:
        """Get the head pointer from the ring buffer state (for debug printing)."""
        return self._state.head

    @property
    def visible(self) -> int:
        """Get the visible count from the ring buffer state (for debug printing)."""
        return self._state.visible

    @property
    def reserved(self) -> int:
        """Get the reserved count from the ring buffer state (for debug printing)."""
        return self._state.reserved

    @property
    def free(self) -> int:
        """Get the free count from the ring buffer state (for debug printing)."""
        return self._state.free()

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

        for block in self._pending_confirmations:
            errors.append(
                f"Block(acquisition={block.acquisition.name}, "
                f"thread={block.thread_type.name}, access={block.access_state.name}): "
                f"block data was used in arithmetic but never reached a store()."
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
            likeness_tensor=self.likeness_tensor,
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
    likeness_tensor: Tensor,
    shape: Shape,
    buffer_factor: Size = 2,
) -> DataflowBuffer:
    """
    Create a DataflowBuffer with the same dtype and element shape as likeness_tensor.

    Args:
        likeness_tensor: A tensor providing dtype and element shape (including degenerate dimensions)
        shape: Tuple of tile-grid dimensions, e.g. (1,) for 1-D, (1, 1) for 2-D,
            (1, 1, 1) for 3-D, etc. The total buffer capacity is
            math.prod(shape) * buffer_factor blocks.
        buffer_factor: Multiplier for total buffer capacity

    Returns:
        A DataflowBuffer with dtype and element shape matching likeness_tensor

    Example:
        x = ttnn.zeros((64, 64), dtype=ttnn.float32)
        x_dfb = make_dataflow_buffer_like(x, shape=(2, 2), buffer_factor=2)
    """
    from .context import get_context

    dfb = DataflowBuffer(
        likeness_tensor=likeness_tensor, shape=shape, buffer_factor=buffer_factor
    )
    ctx = get_context()
    ctx.kernel_dfb_count += 1
    ctx.kernel_l1_bytes += dfb.capacity_bytes
    return dfb


def track_source_blocks(result_block: Block, *input_blocks: Block) -> None:
    """Track source wait() blocks for proper state management.

    Adds input wait() blocks to the result block's _source_blocks list and
    eagerly marks them as read so that pop() can succeed when the block's
    context manager exits, even if store() on the result is called later.

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
            # Fire assign_src so pop() is allowed when the 'with' context exits.
            # MR means the block has not yet been consumed as an arithmetic source.
            # The block is registered as pending store confirmation and cleared
            # by mark_store_read_complete() when store() eventually fires.
            if block.access_state == AccessState.MR:
                block.mark_assign_src_complete()
        elif is_temporary:
            actual_source = getattr(block, "_source_blocks", None)
            result_source = getattr(result_block, "_source_blocks", None)
            if actual_source is not None and result_source is not None:
                result_source.extend(actual_source)


def _matmul_tile_shape(a_shape: Shape, b_shape: Shape) -> Shape:
    """Compute the output tile-grid shape for matmul a @ b.

    Applies PyTorch matmul broadcasting rules in tile-grid space, so the
    declared shape of the operand blocks determines the result shape rather
    than the actual backing-tensor dimensions.

    Examples:
        (1, 1) @ (1, 1)       -> (1, 1)       # standard 2-D
        (1, 1, 1) @ (1, 1)   -> (1, 1, 1)    # batch x 2-D
        (2, 1, 1) @ (2, 1, 1) -> (2, 1, 1)   # same batch
    """
    if len(b_shape) == 1:
        # (..., m, k) @ (k,) -> (..., m)
        return a_shape[:-1]
    if len(a_shape) == 1:
        # (k,) @ (..., k, n) -> (..., n)
        return b_shape[:-2] + (b_shape[-1],)
    if len(a_shape) == 2 and len(b_shape) == 2:
        # Standard 2-D: (m, k) @ (k, n) -> (m, n)
        return (a_shape[0], b_shape[1])
    # Batched: (..., m, k) @ (..., k, n) -> (broadcast_batch, m, n)
    a_batch = a_shape[:-2]
    b_batch = b_shape[:-2]
    if b_batch:
        out_batch: Shape = cast(Shape, tuple(torch.broadcast_shapes(a_batch, b_batch)))  # type: ignore[misc]
    else:
        out_batch = a_batch
    return out_batch + (a_shape[-2], b_shape[-1])


def matmul(a: Block, b: Block, _output_hint: Optional[Block] = None) -> Block:
    """Matrix multiplication of two blocks.

    Converts each block to a ttnnsim.Tensor, delegates to torch.matmul via the
    @ operator, then wraps the result in a Block whose tile-grid shape is
    derived from the declared shapes of the operands (following PyTorch batched
    matmul rules) rather than from the result tensor dimensions.

    Args:
        a: First input block.
        b: Second input block.
        _output_hint: Optional output block hint (unused in simulator).

    Returns:
        Block whose tile shape corresponds to the matmul output shape.
    """
    result_tensor = a.to_tensor() @ b.to_tensor()
    result_shape = _matmul_tile_shape(a.shape, b.shape)
    result_block = Block(
        tensor=result_tensor,
        shape=result_shape,
        acquisition=BlockAcquisition.RESERVE,
        thread_type=ThreadType.COMPUTE,
        is_temporary=True,
    )
    track_source_blocks(result_block, a, b)
    return result_block
