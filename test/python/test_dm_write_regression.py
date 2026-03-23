# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Regression test: tx.wait() in DM write thread for loop.
This pattern is used by every existing tt-lang kernel.
"""

import os
os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl

TILE = 32


@ttl.kernel(grid=(1, 1))
def passthrough_kernel(inp, out):
    """Simplest kernel with a for loop + tx.wait() in DM write."""
    Mt = inp.shape[0] // TILE
    Nt = inp.shape[1] // TILE
    num_tiles = Mt * Nt

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.datamovement()
    def read():
        for tid in range(num_tiles):
            m = tid // Nt
            n = tid % Nt
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp[m, n], blk)
                tx.wait()

    @ttl.compute()
    def compute():
        for _ in range(num_tiles):
            with inp_dfb.wait() as a_blk:
                with out_dfb.reserve() as y_blk:
                    y_blk.store(a_blk)

    @ttl.datamovement()
    def write():
        for tid in range(num_tiles):
            m = tid // Nt
            n = tid % Nt
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[m, n])
                tx.wait()


# Compile test
inp = ttnn.from_torch(
    torch.randn(32, 64, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
)
out = ttnn.from_torch(
    torch.zeros(32, 64, dtype=torch.bfloat16),
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
)
passthrough_kernel(inp, out)
print("PASS: DM write for-loop with tx.wait() compiles")

# CHECK: PASS: DM write for-loop with tx.wait() compiles
