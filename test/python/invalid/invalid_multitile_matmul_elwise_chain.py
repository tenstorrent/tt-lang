# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s > %t.out 2>&1
# RUN: FileCheck %s < %t.out

"""
Multi-tile matmul feeding an elementwise chain (relu(prev + a @ b) with
block_k > 1) is not yet supported. Verify a clear error is emitted.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl

# CHECK: multi-tile matmul feeding an elementwise chain is not yet supported


@ttl.operation(grid=(1, 1))
def matmul_relu_multitile(a, b, acc, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(4, 2), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 4), buffer_factor=2)
    acc_dfb = ttl.make_dataflow_buffer_like(acc, shape=(4, 4), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(4, 4), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            acc_dfb.wait() as prev,
        ):
            with out_dfb.reserve() as o:
                o.store(ttl.math.relu(prev + a_blk @ b_blk))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:4, 0:2], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:2, 0:4], blk)
            tx.wait()
        with acc_dfb.reserve() as blk:
            tx = ttl.copy(acc[0:4, 0:4], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:4, 0:4])
            tx.wait()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_dram

    device = ttnn.open_device(device_id=0)
    try:
        a = to_dram(torch.randn(128, 64, dtype=torch.bfloat16), device)
        b = to_dram(torch.randn(64, 128, dtype=torch.bfloat16), device)
        acc = to_dram(torch.randn(128, 128, dtype=torch.bfloat16), device)
        out = to_dram(torch.zeros(128, 128, dtype=torch.bfloat16), device)
        matmul_relu_multitile(a, b, acc, out)
    finally:
        ttnn.close_device(device)
