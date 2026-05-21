# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: an undefined name on the scalar side of a `scalar * reduce`
expression produces an actionable compiler error pointing at the name, not
an internal AttributeError.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl
from ttlang_test_utils import to_l1


# CHECK: TTLangCompileError
# CHECK: abc
@ttl.operation(grid=(1, 1))
def invalid_reduce_scalar_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            result = abc * ttl.math.reduce_sum(inp_blk, dims=[0, 1])
            out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as inp_blk:
            ttl.copy(inp[0, 0], inp_blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0, 0]).wait()


device = ttnn.open_device(device_id=0)
try:
    inp = to_l1(torch.ones(32, 32, dtype=torch.bfloat16), device)
    out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)
    invalid_reduce_scalar_kernel(inp, out)
finally:
    ttnn.close_device(device)
