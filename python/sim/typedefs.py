# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases with Pydantic constraints for runtime validation.
"""

from typing import Annotated, TypeVar, Tuple, Optional, NamedTuple
from pydantic import Field
from enum import Enum, auto
from dataclasses import dataclass
import torch

CBElemType = TypeVar("CBElemType", int, torch.Tensor)
# Type alias for circular buffer slots
CBSlotType = Optional[CBElemType]


class IndexType(Enum):
    """
    Enumeration of indexing types for TensorAccessors.

    Currently only supports tile-based indexing.
    """

    TILE = auto()


class MulticastType(Enum):
    """
    Enumeration of multicast types.

    PUSH: Data is pushed from source to destinations.
    PULL: Data is pulled by destinations from source.
    """

    PUSH = auto()
    PULL = auto()


PositiveInt = Annotated[int, Field(gt=0)]
NaturalInt = Annotated[int, Field(ge=0)]
Size = PositiveInt
Index = NaturalInt
Count = NaturalInt
CoreIndex = Index


class MulticastAddress(NamedTuple):
    """
    Represents a multicast address for NoC communication.

    Attributes:
        mcast_type: Type of multicast (PUSH or PULL)
        core_indices: Tuple of core indices participating in the multicast
    """

    mcast_type: MulticastType
    core_indices: Tuple[CoreIndex, ...]


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
