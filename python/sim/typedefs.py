# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases with Pydantic constraints for runtime validation.
"""

from typing import Annotated, TypeVar, Tuple, Optional, NamedTuple, Union
from pydantic import Field
from enum import Enum, auto
from dataclasses import dataclass
import torch

CBElemTypeVar = TypeVar("CBElemTypeVar", int, torch.Tensor)
# Type alias for circular buffer slots
CBSlot = Optional[CBElemTypeVar]


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
CoreIndex = Union[Index, Tuple[Index, Index, *tuple[Index, ...]]]


class Pipe(NamedTuple):
    """
    Represents a pipe for NoC communication.

    Attributes:
        src_core: Core index of the source/sender
        dst_core_range: Either a single CoreIndex for unicast, or a Tuple[CoreIndex, CoreIndex]
                       for multicast where the two indices define a rectangular range.
                       Example: ((0, 0), (1, 1)) defines cores (0,0), (0,1), (1,0), (1,1)
    """

    src_core: CoreIndex
    dst_core_range: Union[CoreIndex, Tuple[CoreIndex, CoreIndex]]


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
