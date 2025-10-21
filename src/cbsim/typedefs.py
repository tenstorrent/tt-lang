"""
Type aliases with Pydantic constraints for runtime validation.
"""

from typing import Annotated
from pydantic import Field

MAX_CBS = 32  # Fixed pool of circular buffers

PositiveInt = Annotated[int, Field(gt=0)]
NaturalInt = Annotated[int, Field(ge=0)]
Size = PositiveInt
Index = NaturalInt
Count = NaturalInt
CBID = Annotated[int, Field(ge=0, lt=MAX_CBS)]
