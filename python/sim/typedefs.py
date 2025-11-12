# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type aliases with Pydantic constraints for runtime validation.
"""

from typing import Annotated, TypeVar
from pydantic import Field
import torch
from .constants import MAX_CBS

# Shared TypeVar for generic types across cbapi, cbstate, and ringview
# Constrained to torch.Tensor (specifically intended for float64 dtype tensors)
CBElemType = TypeVar("CBElemType", int, torch.Tensor)

PositiveInt = Annotated[int, Field(gt=0)]
NaturalInt = Annotated[int, Field(ge=0)]
Size = PositiveInt
Index = NaturalInt
Count = NaturalInt
CBID = Annotated[int, Field(ge=0, lt=MAX_CBS)]
