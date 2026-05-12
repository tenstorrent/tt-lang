# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: total CB descriptor size must not exceed per-core L1 CB budget.
"""

# CHECK: exceeds L1 budget
# CHECK: hint:

import ttnn

from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
from ttl.kernel_runner import build_cb_descriptors

# Single compiler-allocated CB: large enough to exceed DEFAULT_L1_CB_BUDGET_BYTES.
# 400 * 2 * 2048 (bf16 tile) = 1,638,400 bytes > 1,466,368.
build_cb_descriptors(
    [None],
    [
        CompilerAllocatedDFBConfig(
            dfb_index=0,
            num_tiles=400,
            data_format="bfloat16",
            block_count=2,
        )
    ],
    None,
)
