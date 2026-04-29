# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file was copied from tt-mlir/tools/pykernel/_src/utils.py
# and cleaned up to remove unused code (_discover_dialect_ops).

import inspect
import re
import textwrap
from typing import Callable

from ttl.dialects import arith
from ttl.ir import *


def _parse_tensor_type(ty_str: str):
    """Extract shape and dtype from an MLIR tiled tensor type."""
    match = re.match(r"tensor<(.+)x!ttcore\.tile<\d+x\d+,\s*(\w+)>>", ty_str)
    if not match:
        return None

    shape_str, dtype = match.groups()
    try:
        shape = tuple(int(dim) for dim in shape_str.split("x"))
    except ValueError:
        return None
    return shape, dtype


def _format_tensor_shape(shape: tuple[int, ...]) -> str:
    return f"({', '.join(str(dim) for dim in shape)})"


def _format_tensor_type(ty_str: str) -> str:
    """Convert MLIR tiled tensor type to user-friendly format.

    Example: tensor<2x2x!ttcore.tile<32x32, bf16>> -> (2, 2) bf16 tensor
    """
    parsed = _parse_tensor_type(ty_str)
    if parsed:
        shape, dtype = parsed
        return f"{_format_tensor_shape(shape)} {dtype} tensor"
    return ty_str


def _tensor_type_mismatch_message(val_type, ty, operation: str = "operation") -> str:
    val_str, ty_str = str(val_type), str(ty)
    if val_str.startswith("tensor<") and ty_str.startswith("tensor<"):
        val_tensor = _parse_tensor_type(val_str)
        ty_tensor = _parse_tensor_type(ty_str)
        if val_tensor and ty_tensor:
            val_shape, val_dtype = val_tensor
            ty_shape, ty_dtype = ty_tensor
            if val_dtype != ty_dtype:
                return (
                    f"incompatible tensor data types for {operation}: got "
                    f"{_format_tensor_type(val_str)} and "
                    f"{_format_tensor_type(ty_str)}; "
                    f"{operation} requires matching data types"
                )
        return (
            f"shape mismatch between {_format_tensor_type(val_str)} and "
            f"{_format_tensor_type(ty_str)}; "
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
