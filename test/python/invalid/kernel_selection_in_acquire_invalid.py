# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject nested selection that cannot execute with its DFB acquisition."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

FAKE_HEADER = "/dev/null/fake_shim.hpp"


# CHECK: statement selects logical kernels (compute, data_movement) outside its enclosing DFB acquire owner (data_movement)
@ttl.operation(grid=(1, 1))
def invalid_nested_selector(inp):
    input_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    with input_dfb.wait() as input_block:
        ttl.copy(input_block, inp[0, 0]).wait()
        ttl.call_extern_func(
            FAKE_HEADER,
            "shared_entry",
            kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
        )


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_nested_selector(inp)
