# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Registry for tensor global names, used to track tensor parameter names."""

from typing import Dict

# Registry mapping tensor id to global name (for tensors that don't support attribute assignment)
_tensor_name_registry: Dict[int, str] = {}


def register_tensor_name(tensor, name: str) -> None:
    """Register a global name for a tensor."""
    _tensor_name_registry[id(tensor)] = name


def get_tensor_global_name(tensor) -> str:
    """Get the global name for a tensor, checking registry first then attribute."""
    tensor_id = id(tensor)
    if tensor_id in _tensor_name_registry:
        return _tensor_name_registry[tensor_id]
    if hasattr(tensor, "_global_name"):
        return tensor._global_name
    raise ValueError("Tensor does not have a global name registered")
