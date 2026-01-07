# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for tt-lang."""

from .block_allocation import split_work_to_cores
from .correctness import assert_allclose, assert_pcc, assert_with_ulp

__all__ = [
    # block_allocation
    "split_work_to_cores",
    # correctness
    "assert_pcc",
    "assert_allclose",
    "assert_with_ulp",
]
