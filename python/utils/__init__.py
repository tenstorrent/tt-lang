# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for tt-lang."""

from .block_allocation import (
    get_large_matmul_params,
    get_number_of_nodes_from_ranges,
    split_work_to_nodes,
)
from .correctness import assert_allclose, assert_pcc, assert_with_ulp

__all__ = [
    # block_allocation
    "split_work_to_nodes",
    "get_number_of_nodes_from_ranges",
    "get_large_matmul_params",
    # correctness
    "assert_pcc",
    "assert_allclose",
    "assert_with_ulp",
]
