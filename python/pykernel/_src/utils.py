# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file was copied from tt-mlir/tools/pykernel/_src/utils.py
# and cleaned up to remove unused code (_discover_dialect_ops).

import textwrap
import inspect
from typing import Callable
from ttmlir.dialects import arith
from ttmlir.ir import *


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
