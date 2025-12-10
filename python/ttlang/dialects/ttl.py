# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTL (TT-Lang) dialect Python bindings."""

# Import C-API bindings from the extension module
from .._mlir_libs._ttlang import ttl as _ttl

# Re-export dialect classes
SliceAttr = _ttl.SliceAttr

__all__ = [
    "SliceAttr",
]
