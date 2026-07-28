# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""Rejects conflicting backing tensor and explicit DFB dtype sources."""

import torch
import ttnn

import ttl


tensor = ttnn.from_torch(
    torch.zeros((32, 32), dtype=torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
)

# CHECK: ValueError: DataflowBuffer dtype
# CHECK-SAME: conflicts with backing tensor dtype
ttl.DataflowBuffer(
    tensor=tensor,
    shape=(1, 1),
    block_count=2,
    dtype=ttnn.float32,
)
