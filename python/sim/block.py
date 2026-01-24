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
    )

    # TODO: We can't do @validate_call here. There reason is that @validate_call actually
    #       copies the arguments to validate them and returns the copies to the decorated
    #       function. In our case, we don't want the copy of the list, we want to use the
    #       original list as is. This is a limitation of pydantic's validate_call, and
    #       perhaps a good reason to look for other frameworks that don't do that! (beartype?)
    # @validate_call
    def __init__(self, buf: List[CBSlot], capacity: Size, span: Span, shape: Shape):
        self._buf = buf
        self._capacity = capacity
        self._span = span
        self._shape = shape
        self._read_locked = False
        self._write_locked = False

    def __len__(self) -> Size:
        return self._span.length

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

    # @validate_call
    def store(self, items: Sequence[Tensor], acc: bool = False) -> None:
        """Store items into the block.

        Args:
            items: Sequence of tensors to store
            acc: If True, accumulate with existing values (+=), otherwise assign (=)
        """
        if len(items) != self._span.length:
            raise ValueError("Length mismatch in store()")
        if acc:
            # Accumulate: add new values to existing values
            for i, v in enumerate(items):
                self._write_slot(i, self[i] + v)
        else:
            # Regular assignment
            for i, v in enumerate(items):
                self._write_slot(i, v)

    def _apply_binary_op(
        self,
        left: Union["Block", List[Tensor]],
        right: Union["Block", List[Tensor]],
        op: Callable[[Any, Any], Any],
    ) -> List[Tensor]:
        """Element-wise binary op: left (op) right with broadcasting support.

        Supports broadcasting when one operand has length 1.
        """
        len_left = len(left)
        len_right = len(right)

        if len_left == len_right:
            # Standard element-wise operation
            return [op(left[i], right[i]) for i in range(len_left)]
        elif len_right == 1:
            # Broadcast right to all elements of left
            right_val = right[0]
            return [op(left[i], right_val) for i in range(len_left)]
        elif len_left == 1:
            # Broadcast left to all elements of right
            left_val = left[0]
            return [op(left_val, right[i]) for i in range(len_right)]
        else:
            raise ValueError(
                f"Operand lengths must match or one must be 1 for broadcasting "
                f"(got lengths {len_left} and {len_right})"
            )

    def _binary_op(
        self,
        other: Union["Block", List[Tensor]],
        op: Callable[[Any, Any], Any],
    ) -> List[Tensor]:
        """Element-wise binary op: self (op) other."""
        return self._apply_binary_op(self, other, op)

    def _rbinary_op(
        self,
        other: List[Tensor],
        op: Callable[[Any, Any], Any],
    ) -> List[Tensor]:
        """Element-wise reverse binary op: other (op) self."""
        return self._apply_binary_op(other, self, op)

    # ---- forward operators ----

    def __add__(self, other: Union["Block", List[Tensor]]) -> List[Tensor]:
        return self._binary_op(other, _op.add)

    def __sub__(self, other: Union["Block", List[Tensor]]) -> List[Tensor]:
        return self._binary_op(other, _op.sub)

    def __mul__(self, other: Union["Block", List[Tensor]]) -> List[Tensor]:
        return self._binary_op(other, _op.mul)

    def __truediv__(self, other: Union["Block", List[Tensor]]) -> List[Tensor]:
        return self._binary_op(other, _op.truediv)

    def __floordiv__(self, other: Union["Block", List[Tensor]]) -> List[Tensor]:
        return self._binary_op(other, _op.floordiv)

    def __mod__(self, other: Union["Block", List[Tensor]]) -> List[Tensor]:
        return self._binary_op(other, _op.mod)

    def __pow__(self, other: Union["Block", List[Tensor]]) -> List[Tensor]:
        return self._binary_op(other, _op.pow)

    def __matmul__(self, other: "Block") -> List[Tensor]:
        return self._binary_op(other, _op.matmul)

    # ---- reverse operators ----

    def __radd__(self, other: List[Tensor]) -> List[Tensor]:
        return self._rbinary_op(other, _op.add)

    def __rsub__(self, other: List[Tensor]) -> List[Tensor]:
        return self._rbinary_op(other, _op.sub)

    def __rmul__(self, other: List[Tensor]) -> List[Tensor]:
        return self._rbinary_op(other, _op.mul)

    def __rtruediv__(self, other: List[Tensor]) -> List[Tensor]:
        return self._rbinary_op(other, _op.truediv)

    def __rfloordiv__(self, other: List[Tensor]) -> List[Tensor]:
        return self._rbinary_op(other, _op.floordiv)

    def __rmod__(self, other: List[Tensor]) -> List[Tensor]:
        return self._rbinary_op(other, _op.mod)

    def __rpow__(self, other: List[Tensor]) -> List[Tensor]:
        return self._rbinary_op(other, _op.pow)

    @property
    def shape(self) -> Shape:
        """Get the shape (rows, cols in tiles) of this block from its associated CB."""
        return self._shape
