# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: only L1 and DRAM memory spaces are supported.

This test verifies that using an invalid memory_space raises ValueError
at decorator time.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: ValueError: Invalid memory_space: 'INVALID'
# CHECK: Must be one of:
@ttl.kernel(grid=(1, 1), memory_space="INVALID")
def invalid_memory_space_kernel(lhs, rhs, out):
    """This kernel should fail because memory_space='INVALID' is not supported."""
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        l = lhs_dfb.wait()
        r = rhs_dfb.wait()
        o = out_dfb.reserve()
        result = l + r
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
    # The error should be raised at module load time (decorator execution)
    # This code should never be reached
    print("ERROR: Expected ValueError was not raised at decorator time!")
    exit(1)
