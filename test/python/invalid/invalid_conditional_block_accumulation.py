# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""
Validation test: conditional block `+=` is rejected by accumulation scope
formation, not by frontend handling for values reassigned across an if
statement.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl
from ttlang_test_utils import to_l1


# CHECK: += inside a conditional is not supported (#504)
@ttl.operation(grid=(1, 1))
def invalid_conditional_block_accumulation(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with out_dfb.reserve() as out_blk:
            for _ in range(2):
                with inp_dfb.wait() as inp_blk:
                    if 1:
                        out_blk += inp_blk

    @ttl.datamovement()
    def reader():
        for _ in range(2):
            with inp_dfb.reserve() as inp_blk:
                ttl.copy(inp[0, 0], inp_blk).wait()

    @ttl.datamovement()
    def writer():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0, 0]).wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        inp = to_l1(torch.ones((32, 32), dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)
        invalid_conditional_block_accumulation(inp, out)
    finally:
        ttnn.close_device(device)
