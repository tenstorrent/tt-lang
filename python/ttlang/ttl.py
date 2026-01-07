# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTL DSL module providing decorator APIs (@ttl.kernel(), @ttl.compute(), @ttl.datamovement())
"""

from ttlang.ttl_api import compute, datamovement
from ttlang.ttl_api import pykernel_gen as kernel

__all__ = [
    "kernel",
    "compute",
    "datamovement",
]
