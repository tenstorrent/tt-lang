# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: matmul output exceeding DST capacity is rejected.

A 4x4 tile matmul produces 16 output tiles, which exceeds the bf16 DST
capacity of 8. The compiler must reject this with a clear error rather
than silently generating incorrect code via subblocking.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: matmul output 4x4 = 16 tiles exceeds DST capacity of 8
@ttl.kernel(grid=(1, 1))
def matmul_exceeds_dst(a, b, y):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(4, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 4), buffer_factor=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(4, 4), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
            with y_dfb.reserve() as y_blk:
                y_blk.store(a_blk @ b_blk)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:4, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0:4], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with y_dfb.wait() as blk:
            tx = ttl.copy(blk, y[0:4, 0:4])
            tx.wait()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_dram

    device = ttnn.open_device(device_id=0)
    try:
        make = lambda: to_dram(torch.randn(128, 128, dtype=torch.bfloat16), device)
        matmul_exceeds_dst(make(), make(), make())

        print("ERROR: Expected error was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
