# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""
Validation test: plain Python values reassigned in an if statement are rejected
with a source-level diagnostic.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl


def _host_ttnn(shape):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


# CHECK: Variable 'tile_count' is reassigned inside an if statement, but it is a plain Python value, such as a tuple, list, string, or integer; TT-Lang only supports reassigning TT-Lang tensor, block, and scalar values across an if statement; move the Python assignment outside the if statement or use a different local variable name inside the branch
@ttl.operation(grid=(1, 1))
def invalid_if_carried_python_value(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        tile_count = (1, 1)
        if 1:
            tile_count = (2, 2)
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            out_blk.store(ttl.block.fill(inp_blk, 0))

    @ttl.datamovement()
    def reader():
        with inp_dfb.reserve() as inp_blk:
            ttl.copy(inp[0, 0], inp_blk).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0, 0]).wait()


if __name__ == "__main__":
    inp = _host_ttnn((32, 32))
    out = _host_ttnn((32, 32))
    invalid_if_carried_python_value(inp, out)
