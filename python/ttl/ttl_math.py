# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL math operations namespace (ttl.math).

Re-exports elementwise operations from the generated module.
"""

# Re-export all generated elementwise operations
from ._generated_elementwise import *  # noqa: F401,F403
from ._generated_elementwise import __all__ as _generated_all
from .operators import (
    broadcast,
    exp,
    fill,
    reduce_max,
    reduce_sum,
    transpose,
    typecast,
)


__all__ = [
    "broadcast",
    "exp",
    "fill",
    "reduce_max",
    "reduce_sum",
    "transpose",
    "typecast",
    *_generated_all,
]
