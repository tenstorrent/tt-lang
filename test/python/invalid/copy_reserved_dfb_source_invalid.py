# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject a reserve-acquired DFB block used as a tensor copy source."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


@ttl.operation(grid=(1, 1))
def copy_reserved_dfb_source(input_tensor, output_tensor):
    scratch_dfb = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=1,
    )
    with scratch_dfb.reserve() as scratch_block:
        ttl.copy(input_tensor[0, 0], scratch_block).wait()
        # CHECK: error: copy() from a DFB block to a tensor requires a block acquired from wait(), not reserve()
        # CHECK: copy_reserved_dfb_source_invalid.py:[[#@LINE+1]]:9
        ttl.copy(scratch_block, output_tensor[0, 0]).wait()


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    output_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    copy_reserved_dfb_source(input_tensor, output_tensor)
