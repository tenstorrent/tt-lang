# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""Validation test: ttl.math.exp rejects unsupported iteration counts."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

from ttlang_test_utils import to_l1


# CHECK: ttl.math.exp iterations must be 8, got 4
@ttl.operation(grid=(1, 1))
def invalid_exp_iterations_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            out_blk.store(ttl.math.exp(inp_blk, iterations=4))

    @ttl.datamovement()
    def dm_read():
        ttl.copy(inp[0, 0], inp_dfb.reserve()).wait()

    @ttl.datamovement()
    def dm_write():
        ttl.copy(out_dfb.wait(), out[0, 0]).wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        inp = to_l1(torch.ones(32, 32, dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)
        invalid_exp_iterations_kernel(inp, out)
    finally:
        ttnn.close_device(device)
