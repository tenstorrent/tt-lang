# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""
Validation test: push() and pop() require blocks, not regular tensors.

This test verifies that calling push()/pop() on non-block tensors
raises the expected ValueError during kernel compilation.

This test uses torch tensors directly to work in COMPILE_ONLY mode
without requiring device hardware.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl


# CHECK: error: push() must be called on a block acquired from reserve(), not a regular tensor
# CHECK-NEXT:   --> {{.*}}invalid_push_pop_on_tensor.py:41:{{[0-9]+}}
@ttl.kernel(grid=(1, 1))
def invalid_push_kernel(lhs, rhs, out):
    """This kernel should fail because push() is called on a non-block tensor."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        result = l + r
        # INVALID: Trying to push() a computed result (not a block)
        result.push()
        o.store(result)
        l.pop()
        r.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        tx_lhs = ttl.copy(lhs[0, 0], lhs_blk)
        tx_lhs.wait()
        lhs_blk.push()

        rhs_blk = rhs_dfb.reserve()
        tx_rhs = ttl.copy(rhs[0, 0], rhs_blk)
        tx_rhs.wait()
        rhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


if __name__ == "__main__":
    lhs_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    rhs_torch = torch.zeros((32, 32), dtype=torch.bfloat16)
    out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

    # This should raise ValueError during compilation
    invalid_push_kernel(lhs_torch, rhs_torch, out_torch)

    print("ERROR: Expected ValueError was not raised!")
    exit(1)
