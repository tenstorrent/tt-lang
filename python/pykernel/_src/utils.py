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

from ttmlir.dialects import arith
from ttmlir.ir import *


def _format_tensor_type(ty_str: str) -> str:
    """Convert MLIR tensor type to user-friendly format.

    Example: tensor<2x2x!ttcore.tile<32x32, bf16>> -> (2, 2) bf16 tensor
    """
    match = re.match(r"tensor<(\d+)x(\d+)x!ttcore\.tile<\d+x\d+,\s*(\w+)>>", ty_str)
    if match:
        rows, cols, dtype = match.groups()
        return f"({rows}, {cols}) {dtype} tensor"
    return ty_str


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
        # Check for tensor shape mismatch and provide helpful error
        val_str, ty_str = str(val.type), str(ty)
        if val_str.startswith("tensor<") and ty_str.startswith("tensor<"):
            raise TypeError(
                f"shape mismatch between {_format_tensor_type(val_str)} and "
                f"{_format_tensor_type(ty_str)}; "
                f"note: you can use ttl.math.broadcast() to expand the smaller tensor"
            )
        raise TypeError(f"Unhandled cast from {val.type} to {ty}")


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
