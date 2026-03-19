# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: fusing matmul result with elementwise add in a single
store expression is not yet supported. The user must store the matmul
result to a temporary CB first, then add in a separate compute step.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: fusion failed: cannot trace through non-elementwise op
@ttl.kernel(grid=(1, 1))
def invalid_matmul_fusion(a, b, c, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, c_dfb.wait() as c_blk:
            with out_dfb.reserve() as o:
                o.store((a_blk @ b_blk) + c_blk)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(c[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_dram

    device = ttnn.open_device(device_id=0)
    try:
        make = lambda: to_dram(torch.randn(32, 32, dtype=torch.bfloat16), device)
        invalid_matmul_fusion(make(), make(), make(), make())
        print("ERROR: Expected error was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
