# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""All-gather minimal matmul operation and runnable example."""

from .config import AllGatherMinimalMatmulConfig
from .operation import make_all_gather_minimal_matmul_operation

__all__ = [
    "AllGatherMinimalMatmulConfig",
    "make_all_gather_minimal_matmul_operation",
]
