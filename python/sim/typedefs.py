# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases with Pydantic constraints for runtime validation.
"""

from typing import Annotated, TypeVar
from pydantic import Field
from enum import Enum, auto
import torch
from .constants import MAX_CBS

CBElemType = TypeVar("CBElemType", int, torch.Tensor)


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
CBID = Annotated[NaturalInt, Field(ge=0, lt=MAX_CBS)]
