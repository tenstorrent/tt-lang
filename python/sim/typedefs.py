# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases with Pydantic constraints for runtime validation.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Annotated, Generic, Tuple, TypeVar, Union

from pydantic import Field


# TODO: Expand IndexType as needed, see relevant issue:
#       https://github.com/tenstorrent/tt-lang/issues/69
class IndexType(Enum):
    """
    Enumeration of indexing types for TensorAccessors.

    Currently only supports tile-based indexing.
    """

    TILE = auto()


PositiveInt = Annotated[int, Field(gt=0)]
NaturalInt = Annotated[int, Field(ge=0)]
Size = PositiveInt
Index = NaturalInt
Count = NaturalInt
CoreCoord = Union[Index, Tuple[Index, ...]]
CoreRange = Tuple[Union[Index, slice], ...]

# Type variable for Pipe destination type
DstT = TypeVar("DstT", CoreCoord, CoreRange, Tuple[CoreCoord, CoreCoord])


@dataclass(frozen=True)
class Pipe(Generic[DstT]):
    """
    Represents a pipe for NoC communication.

    A pipe describes a data transfer from a source core to destination core(s).
    Can be used for both unicast (single destination) and multicast (multiple destinations).

    Type Parameters:
        DstT: The type of the destination - CoreCoord, CoreRange, or Tuple[CoreCoord, CoreCoord]

    Attributes:
        src_core: Core coordinates of the source/sender. Can be:
                 - Index: Single 1D core (e.g., 0, 1, 2)
                 - Tuple[Index, ...]: Multi-dimensional core (e.g., (0, 1), (1, 2, 3))

        dst_core_range: Destination specification. Can be:
                       - CoreCoord: Single destination core (unicast)
                         Example: 5 or (1, 2)
                       - CoreRange: Range of destination cores using slices (multicast)
                         Example: (0, slice(1, 4)) means cores (0,1), (0,2), (0,3)
                       - Tuple[CoreCoord, CoreCoord]: Legacy rectangular range (deprecated, use CoreRange instead)
                         Example: ((0, 0), (1, 1)) means cores (0,0), (0,1), (1,0), (1,1)
    """

    src_core: CoreCoord
    dst_core_range: DstT

    def has_current_core(self) -> bool:
        """Check if the current core participates in this pipe (either as source or destination).

        This is useful for early-exit patterns where non-participating cores should skip work.
        Must be called within a kernel context.

        Returns:
            True if the current core is either the source or in the destination range.
        """
        from .kernel import core, flatten_core_index

        # Check if current core is the source
        current_core_linear = core(dims=1)
        pipe_src_linear = flatten_core_index(self.src_core)
        if current_core_linear == pipe_src_linear:
            return True

        # Check if current core is in destination range
        from .pipe import _core_in_dst_range

        return _core_in_dst_range(self.dst_core_range)

    def __hash__(self) -> int:
        """Custom hash implementation to handle slices and nested tuples."""

        def make_hashable(obj: Any) -> Any:
            """Convert potentially unhashable objects to hashable equivalents."""
            if isinstance(obj, slice):
                return (obj.start, obj.stop, obj.step)  # type: ignore[return-value]
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)  # type: ignore[misc]
            elif isinstance(obj, tuple):
                return tuple(make_hashable(item) for item in obj)  # type: ignore[misc]
            else:
                return obj

        return hash((make_hashable(self.src_core), make_hashable(self.dst_core_range)))


Shape = Tuple[Size, ...]
_MAX_CBS: Size = 32  # Fixed pool of circular buffers
CBID = Annotated[NaturalInt, Field(ge=0, lt=_MAX_CBS)]


@dataclass(frozen=True)
class Span:
    """A span representing a contiguous range in a ring buffer.

    Attributes:
        start: Inclusive index in underlying ring
        length: Number of tiles
    """

    start: Index  # inclusive index in underlying ring
    length: Size  # number of tiles
