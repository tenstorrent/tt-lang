# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Block and supporting Span for cbsim.
"""

import operator as _op
from typing import Any, Callable, List, Sequence, Union

from pydantic import validate_call

from .cbstate import CBSlot
from .ttnnsim import Tensor
from .typedefs import Index, Shape, Size, Span


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
    """

    __slots__ = (
        "_buf",
        "_capacity",
        "_span",
        "_shape",
        "_read_locked",
        "_write_locked",
        "_is_temporary",
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
        is_temporary: bool = False,
    ):
        self._buf = buf
        self._capacity = capacity
        self._span = span
        self._shape = shape
        self._read_locked = False
        self._write_locked = False
        self._is_temporary = is_temporary

    @classmethod
    def from_list(cls, tensors: List[Tensor], shape: Shape) -> "Block":
        """Create a temporary Block from a list of tensors (computation result).

        Temporary blocks are not backed by CB storage and don't support wrap-around.
        """
        return cls(
            buf=tensors,
            capacity=len(tensors),
            span=Span(0, len(tensors)),
            shape=shape,
            is_temporary=True,
        )

    def __len__(self) -> Size:
        return self._span.length

    @property
    def is_temporary(self) -> bool:
        """Check if this Block is a temporary computation result (not CB-backed)."""
        return self._is_temporary

    def _check_can_read(self) -> None:
        """Check if this Block can be read from.

        Raises:
            RuntimeError: If Block is locked for writing by an active copy
        """
        if self._write_locked:
            raise RuntimeError(
                "Cannot read from Block: locked as copy destination until wait() completes"
            )

    def _check_can_write(self) -> None:
        """Check if this Block can be written to.

        Raises:
            RuntimeError: If Block is locked for reading or writing by an active copy
        """
        if self._read_locked:
            raise RuntimeError(
                "Cannot write to Block: locked as copy source until wait() completes"
            )
        if self._write_locked:
            raise RuntimeError(
                "Cannot write to Block: locked as copy destination until wait() completes"
            )

    def lock_for_read(self) -> None:
        """Lock this Block for reading (used as copy source).

        Raises:
            RuntimeError: If Block is already locked for writing
        """
        if self._write_locked:
            raise RuntimeError(
                "Cannot use Block as copy source: locked as copy destination until wait() completes"
            )
        self._read_locked = True

    def lock_for_write(self) -> None:
        """Lock this Block for writing (used as copy destination).

        Raises:
            RuntimeError: If Block is already locked for reading or writing
        """
        if self._read_locked:
            raise RuntimeError(
                "Cannot use Block as copy destination: locked as copy source until wait() completes"
            )
        if self._write_locked:
            raise RuntimeError(
                "Cannot use Block as copy destination: already locked as copy destination until wait() completes"
            )
        self._write_locked = True

    def unlock_read(self) -> None:
        """Unlock this Block from reading."""
        self._read_locked = False

    def unlock_write(self) -> None:
        """Unlock this Block from writing."""
        self._write_locked = False

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
        """Set item with lock checking."""
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
        """
        # Convert Block to sequence if needed
        if isinstance(items, Block):
            items_seq = items.to_list()
        else:
            items_seq = items

        if len(items_seq) != self._span.length:
            raise ValueError("Length mismatch in store()")
        if acc:
            # Accumulate: add new values to existing values
            for i, v in enumerate(items_seq):
                self[i] = self[i] + v
        else:
            # Regular assignment
            for i, v in enumerate(items_seq):
                self[i] = v

    def _apply_binary_op(
        self,
        left: "Block",
        right: "Block",
        op: Callable[[Any, Any], Any],
    ) -> List[Tensor]:
        """Element-wise binary op: left (op) right with broadcasting support.

        Supports broadcasting using PyTorch's native broadcasting rules.
        """
        len_left = len(left)
        len_right = len(right)

        # Simple cases: equal length or scalar broadcasting
        if len_left == len_right:
            return [op(left[i], right[i]) for i in range(len_left)]
        elif len_right == 1:
            right_val = right[0]
            return [op(left[i], right_val) for i in range(len_left)]
        elif len_left == 1:
            left_val = left[0]
            return [op(left_val, right[i]) for i in range(len_right)]

        # Both operands are Blocks with shapes - delegate to ttnnsim for broadcasting
        from .ttnnsim import broadcast_tensors

        return broadcast_tensors(list(left), list(right), left._shape, right._shape, op)

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

    def __pow__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.pow)

    def __matmul__(self, other: "Block") -> "Block":
        return self._binary_op(other, _op.matmul)

    @property
    def shape(self) -> Shape:
        """Get the shape (rows, cols in tiles) of this block from its associated CB."""
        return self._shape
