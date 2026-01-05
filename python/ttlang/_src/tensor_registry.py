# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Registry for tensor global names, used to track tensor parameter names."""

from typing import Dict, Optional, Tuple

# Registry mapping tensor id to global name (for tensors that don't support attribute assignment)
_tensor_name_registry: Dict[int, str] = {}
# Registry mapping tensor id to global index
_tensor_index_registry: Dict[int, int] = {}
# Registry mapping tensor id to source location (file, line)
_tensor_source_registry: Dict[int, Tuple[str, int]] = {}


def register_tensor_name(tensor, name: str, index: int = -1) -> None:
    """Register a global name and index for a tensor."""
    _tensor_name_registry[id(tensor)] = name
    if index >= 0:
        _tensor_index_registry[id(tensor)] = index


def get_tensor_global_index(tensor) -> int:
    """Get the global index for a tensor."""
    tensor_id = id(tensor)
    if tensor_id in _tensor_index_registry:
        return _tensor_index_registry[tensor_id]
    raise ValueError("Tensor does not have a global index registered")


def get_tensor_global_name(tensor) -> str:
    """Get the global name for a tensor, checking registry first then attribute."""
    tensor_id = id(tensor)
    if tensor_id in _tensor_name_registry:
        return _tensor_name_registry[tensor_id]
    if hasattr(tensor, "_global_name"):
        return tensor._global_name
    raise ValueError("Tensor does not have a global name registered")


def register_tensor_source(tensor, source_file: str, line: int) -> None:
    """Register the source location where a tensor variable was assigned."""
    _tensor_source_registry[id(tensor)] = (source_file, line)


def get_tensor_source(tensor) -> Optional[Tuple[str, int]]:
    """Get the source location where a tensor was assigned, if tracked."""
    return _tensor_source_registry.get(id(tensor))
