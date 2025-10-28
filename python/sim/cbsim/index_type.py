"""
IndexType enumeration for specifying how streams index into tensors.
"""

from enum import Enum, auto


class IndexType(Enum):
    """
    Enumeration of indexing types for stream accessors.
    
    Currently only supports tile-based indexing.
    """
    
    TILE = auto()
