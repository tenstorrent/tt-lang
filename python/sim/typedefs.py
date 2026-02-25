# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases with Pydantic constraints for runtime validation.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Annotated, Tuple, Union

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

# A single dimension selector: either a non-negative integer coordinate or a
# slice range.  Used for core ranges and tensor tile-coordinate keys.
Selector = Union[Index, slice]

CoreRange = Tuple[Selector, ...]

Shape = Tuple[Size, ...]

# Valid key type for Tensor.__getitem__ / __setitem__: a tuple of 2 to
# MAX_TENSOR_DIMS elements, each an int (single tile coordinate) or slice
# (tile-coordinate range).  The last two elements index the tile row and
# tile column; preceding elements are batch indices (implicit tile size 1,
# so tile-space and element-space are identical for those dimensions).
TensorKey = Tuple[Selector, ...]
_MAX_DFBS: Size = 32  # Fixed pool of circular buffers
DFBID = Annotated[NaturalInt, Field(ge=0, lt=_MAX_DFBS)]


@dataclass(frozen=True)
class Span:
    """A span representing a contiguous range in a ring buffer.

    Attributes:
        start: Inclusive index in underlying ring
        length: Number of tiles
    """

    start: Index  # inclusive index in underlying ring
    length: Size  # number of tiles
