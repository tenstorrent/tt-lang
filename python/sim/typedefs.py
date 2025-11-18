# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases with Pydantic constraints for runtime validation.
"""

from typing import Annotated, TypeVar, Tuple, Optional
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


PositiveInt = Annotated[int, Field(gt=0)]
NaturalInt = Annotated[int, Field(ge=0)]
Size = PositiveInt
Index = NaturalInt
Count = NaturalInt
CoreIndex = Index
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
