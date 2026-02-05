# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: binary ops on incompatible CB shapes.

Trying to add tensors with mismatched CB shapes (e.g., (2,1) + (2,2))
should produce an error suggesting broadcast.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: shape mismatch between (2, 2) bf16 tensor and (2, 1) bf16 tensor; note: you can use ttl.math.broadcast() to expand the smaller tensor
@ttl.kernel(grid=(1, 1))
def mismatched_shape_kernel(a, b, out):
    """INVALID: add tensors with mismatched CB shapes."""
    # a_cb is (2, 1) - column vector
    a_cb = ttl.make_circular_buffer_like(a, shape=(2, 1), buffer_factor=2)
    # b_cb is (2, 2) - full grid
    b_cb = ttl.make_circular_buffer_like(b, shape=(2, 2), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(2, 2), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, out_cb.reserve() as o:
            # This should fail - adding (2,1) to (2,2) without broadcast
            result = a_tile + b_tile
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        with a_cb.reserve() as a_blk:
            tx_a = ttl.copy(a[0:2, 0:1], a_blk)
            tx_a.wait()
        with b_cb.reserve() as b_blk:
            tx_b = ttl.copy(b[0:2, 0:2], b_blk)
            tx_b.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0:2, 0:2])
            tx.wait()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_l1

    device = ttnn.open_device(device_id=0)

    try:
        a = to_l1(torch.ones((64, 32), dtype=torch.bfloat16), device)
        b = to_l1(torch.ones((64, 64), dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros((64, 64), dtype=torch.bfloat16), device)

        mismatched_shape_kernel(a, b, out)

        print("ERROR: Expected error was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
