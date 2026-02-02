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
from .operators import broadcast, matmul, reduce_sum, reduce_max

__all__ = [
    "broadcast",
    "matmul",
    "reduce_sum",
    "reduce_max",
    *_generated_all,
]
