# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject logical-kernel selection on a DFB acquisition."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


# CHECK: kernel= is not supported on DFB reserve(); select release ownership on push() or pop()
@ttl.operation(grid=(1, 1))
def invalid_acquire_selector(inp):
    output_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    output_dfb.reserve(kernel=ttl.KernelKind.COMPUTE)


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_acquire_selector(inp)
