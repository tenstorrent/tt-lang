"""
IndexType enumeration for specifying how TensorAccessors index into tensors.
"""

from enum import Enum, auto


class IndexType(Enum):
    """
    Enumeration of indexing types for TensorAccessors.
    
    Currently only supports tile-based indexing.
    """
    
    TILE = auto()
