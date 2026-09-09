# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""All-gather minimal matmul operation and runnable example."""

from .operation import (
    AllGatherMinimalMatmulConfig,
    make_all_gather_minimal_matmul_operation,
)

__all__ = [
    "AllGatherMinimalMatmulConfig",
    "make_all_gather_minimal_matmul_operation",
]
