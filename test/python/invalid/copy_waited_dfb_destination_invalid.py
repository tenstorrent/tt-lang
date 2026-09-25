# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject a wait-acquired DFB block used as a tensor copy destination."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


@ttl.operation(grid=(1, 1))
def copy_waited_dfb_destination(input_tensor):
    scratch_dfb = ttl.make_dataflow_buffer_like(
        input_tensor,
        shape=(1, 1),
        block_count=1,
    )
    with scratch_dfb.wait() as scratch_block:
        # CHECK: error: copy() from a tensor to a DFB block requires a block acquired from reserve(), not wait()
        # CHECK: copy_waited_dfb_destination_invalid.py:[[#@LINE+1]]:9
        ttl.copy(input_tensor[0, 0], scratch_block).wait()


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    copy_waited_dfb_destination(input_tensor)
