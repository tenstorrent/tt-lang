# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Block namespace for shape-manipulation and utility operations.

Specification revision 0.17 places these operations under ``ttl.block``. Inside
a kernel the AST compiler resolves them by name, so this module exists for the
uses that Python itself must resolve: importing, aliasing, and introspecting a
signature outside kernel source.

Only the operations the compiler implements appear here. ``mask``,
``mask_posinf``, ``where``, ``squeeze`` and ``unsqueeze`` are specified but not
yet implemented, and are absent rather than bound to a stub.
"""

from .operators import broadcast, fill, transpose

__all__ = [
    "broadcast",
    "fill",
    "transpose",
]
