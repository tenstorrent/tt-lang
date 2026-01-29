# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: bcast input must be CB-attached, not an elementwise result.

Bcast reads directly from CB, so its input cannot be a DST value (like an
elementwise result). The DSL catches this early with a clear error.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: broadcast input must come directly from a circular buffer
# CHECK:   --> {{.*}}invalid_bcast_not_first.py:[[LINE:[0-9]+]]:{{[0-9]+}}
# CHECK:    |
# CHECK: [[LINE]] |             result = ttl.math.broadcast(ab, o, dims=[0])
# CHECK:    |                      ^
@ttl.kernel(grid=(1, 1))
def invalid_bcast_kernel(a, b, out):
    """INVALID: bcast on elementwise result (not CB-attached)."""
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with a_cb.wait() as a_tile, b_cb.wait() as b_tile, out_cb.reserve() as o:
            ab = a_tile * b_tile
            result = ttl.math.broadcast(ab, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        tx_a = ttl.copy(a[0, 0], a_cb.reserve())
        tx_a.wait()
        a_cb.push()
        tx_b = ttl.copy(b[0, 0], b_cb.reserve())
        tx_b.wait()
        b_cb.push()

    @ttl.datamovement()
    def dm_write():
        tx = ttl.copy(out_cb.wait(), out[0, 0])
        tx.wait()
        out_cb.pop()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_l1

    device = ttnn.open_device(device_id=0)

    try:
        a = to_l1(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
        b = to_l1(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

        invalid_bcast_kernel(a, b, out)

        print("ERROR: Expected error was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
