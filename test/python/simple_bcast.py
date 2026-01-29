# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s < %t.output.txt

# CHECK: PASS

"""Fused bcast test: bcast(c) + (a * b) in single compute block."""

import torch
import ttnn
import ttl
from ttlang_test_utils import to_l1


@ttl.kernel(grid=(1, 1))
def fused_bcast_kernel(a, b, c, out):
    """Compute bcast(c) + (a * b) in a single fused compute block.

    Bcast must be first operation - it reads from CB and writes to DST.
    Subsequent elementwise ops read from DST.
    """
    a_cb = ttl.make_circular_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_cb = ttl.make_circular_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_cb = ttl.make_circular_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        # Single fused block: bcast first, then elementwise
        with (
            a_cb.wait() as a_tile,
            b_cb.wait() as b_tile,
            c_cb.wait() as c_tile,
            out_cb.reserve() as o,
        ):
            c_bcast = ttl.math.broadcast(c_tile, o, dims=[0])
            ab = a_tile * b_tile
            result = c_bcast + ab
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        a_blk = a_cb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_cb.push()

        b_blk = b_cb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_cb.push()

        c_blk = c_cb.reserve()
        tx_c = ttl.copy(c[0, 0], c_blk)
        tx_c.wait()
        c_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()


def main():
    device = ttnn.open_device(device_id=0)

    # c = row tile with first row = 1.0 (for row broadcast)
    c_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    c_torch[0, :] = 1.0

    a = to_l1(torch.full((32, 32), 2.0, dtype=torch.bfloat16), device)
    b = to_l1(torch.full((32, 32), 3.0, dtype=torch.bfloat16), device)
    c = to_l1(c_torch, device)
    out = to_l1(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    fused_bcast_kernel(a, b, c, out)

    result = ttnn.to_torch(out)
    # Expected: (2.0 * 3.0) + 1.0 = 7.0
    expected = torch.full((32, 32), 7.0, dtype=torch.bfloat16)

    print(f"Result unique values: {torch.unique(result).tolist()}")
    print(f"Expected: all 7.0")

    if torch.allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2):
        print("PASS")
    else:
        print("FAIL")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
