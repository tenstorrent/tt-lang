# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTL (TT-Lang) dialect Python bindings."""

from ttmlir._mlir_libs import get_dialect_registry

from .._mlir_libs import _ttlang
from .._mlir_libs._ttlang import ttl_ir as ir
from ._ttl_enum_gen import *  # noqa: F401,F403
from ._ttl_ops_gen import *  # noqa: F401,F403


def ensure_dialects_registered(ctx):
    """Ensure TTL dialect is registered with the given MLIR context."""
    reg = get_dialect_registry()
    _ttlang.register_dialects(reg)
    ctx.append_dialect_registry(reg)
    # Trigger loading so attributes/ops are available immediately.
    _ = ctx.dialects["ttl"]


# Re-export C++-bound attributes/types for convenience.
SliceAttr = ir.SliceAttr
CircularBufferType = ir.CircularBufferType

__all__ = [  # noqa: F405
    *[name for name in globals().keys() if not name.startswith("_")],
]
