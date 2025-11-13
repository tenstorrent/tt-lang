# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
_RingView and supporting Span for cbsim.
"""

from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence
from .typedefs import Size, Index, CBElemType
from pydantic import validate_call


# Notice that get_read_ptr and get_write_ptr return a C++ pointer which does not
# necessarily make sense in a python context. So we need something that can
# access the elements of the cb (as a pointer would) from the position the
# pointer points. To hide needless index arithmetic, we also add the ability to
# wrap around. Notice also that it handles a list and a capacity, instead of a
# _CBState, a deliberate choice to make it closer in spirit to a pointer and
# minimizing the state that is exposed.
@dataclass(frozen=True)
class Span:
    start: Index  # inclusive index in underlying ring
    length: Size  # number of tiles


class RingView(Generic[CBElemType]):
    """A logically contiguous window into the ring, possibly wrapping.
    Provides list-like access to elements while respecting wrap-around.
    """

    __slots__ = ("_buf", "_capacity", "_span")

    # TODO: We can't do @validate_call here. There reason is that @validate_call actually
    #       copies the arguments to validate them and returns the copies to the decorated
    #       function. In our case, we don't want the copy of the list, we want to use the
    #       original list as is. This is a limitation of pydantic's validate_call, and
    #       perhaps a good reason to look for other frameworks that don't do that! (beartype?)
    # @validate_call
    def __init__(self, buf: List[Optional[CBElemType]], capacity: Size, span: Span):
        self._buf = buf
        self._capacity = capacity
        self._span = span

    def __len__(self) -> Size:
        return self._span.length

    @validate_call
    def __getitem__(self, idx: Index) -> CBElemType:
        if not (0 <= idx < self._span.length):
            raise IndexError(idx)
        value = self._buf[(self._span.start + idx) % self._capacity]
        if value is None:
            raise ValueError(f"Reading uninitialized or consumed slot at index {idx}")
        return value

    # TODO: Why does validate_call fail here? Maybe because CBElemType could
    # resolve to tensor which is similar to a list?
    # @validate_call
    def __setitem__(self, idx: Index, value: CBElemType) -> None:
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

    def to_list(self) -> List[Optional[CBElemType]]:
        return [self[i] for i in range(len(self))]

    # @validate_call
    def store(self, items: Sequence[CBElemType]) -> None:
        if len(items) != self._span.length:
            raise ValueError("Length mismatch in store()")
        for i, v in enumerate(items):
            self[i] = v
