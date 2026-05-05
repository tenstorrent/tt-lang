# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file was copied from tt-mlir/tools/pykernel/_src/utils.py
# and cleaned up to remove unused code (_discover_dialect_ops).

import inspect
import textwrap
from typing import Callable

from ttl.dialects import arith
from ttl.ir import *


def _as_ranked_tensor_type(ty):
    if isinstance(ty, RankedTensorType):
        return ty

    maybe_downcast = getattr(RankedTensorType, "maybe_downcast", None)
    if maybe_downcast is None:
        return None
    return maybe_downcast(ty)


def _format_tensor_shape(shape: tuple[int, ...]) -> str:
    return f"({', '.join(str(dim) for dim in shape)})"


def _format_element_type(element_type) -> str:
    element_type_str = str(element_type)
    if element_type_str.startswith("!ttcore.tile<") and "," in element_type_str:
        return element_type_str.rsplit(",", 1)[1].rstrip(">").strip()
    return element_type_str


def _format_tensor_type(ty) -> str:
    """Convert MLIR tiled tensor type to user-friendly format.

    Example: tensor<2x2x!ttcore.tile<32x32, bf16>> -> (2, 2) bf16 tensor
    """
    tensor_type = _as_ranked_tensor_type(ty)
    if tensor_type is not None:
        return (
            f"{_format_tensor_shape(tuple(tensor_type.shape))} "
            f"{_format_element_type(tensor_type.element_type)} tensor"
        )
    return str(ty)


def _tensor_type_mismatch_message(val_type, ty, operation: str = "operation") -> str:
    val_tensor = _as_ranked_tensor_type(val_type)
    ty_tensor = _as_ranked_tensor_type(ty)
    if val_tensor is not None and ty_tensor is not None:
        if val_tensor.element_type != ty_tensor.element_type:
            return (
                f"incompatible tensor data types for {operation}: got "
                f"{_format_tensor_type(val_tensor)} and "
                f"{_format_tensor_type(ty_tensor)}; "
                f"{operation} requires matching data types"
            )
        return (
            f"shape mismatch between {_format_tensor_type(val_tensor)} and "
            f"{_format_tensor_type(ty_tensor)}; "
            f"note: you can use ttl.math.broadcast() to expand the smaller tensor"
        )
    return f"Unhandled cast from {val_type} to {ty}"


def _cleanup_source_code(f: Callable):
    source_code = inspect.getsource(f)
    source_code = textwrap.dedent(source_code)
    cleaned = [
        line for line in source_code.splitlines() if not line.strip().startswith("@")
    ]
    source_code = "\n".join(cleaned)
    return source_code


def _cast(val, ty):
    if val.type == ty or (isinstance(ty, type) and isinstance(val.type, ty)):
        return val

    if ty is IndexType or isinstance(ty, IndexType):
        return arith.index_cast(IndexType.get(), val)
    elif isinstance(val.type, IndexType) and isinstance(ty, IntegerType):
        return arith.index_cast(ty, val)
    else:
        # Check for tensor mismatches and provide helpful errors.
        raise TypeError(_tensor_type_mismatch_message(val.type, ty))


def _asindex(val):
    if val is None:
        return val
    if isinstance(val, tuple):
        return tuple(map(_asindex, val))
    if isinstance(val, list):
        return list(map(_asindex, val))
    return _cast(val, IndexType)


def _get_type_str(ty):
    s = str(ty).split("<")[0]
    if not s.startswith("!"):
        s = "!" + s
    return s
