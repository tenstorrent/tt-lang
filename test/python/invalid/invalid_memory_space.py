# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: only L1 and DRAM memory spaces are supported.

This test verifies that using an invalid memory_space raises ValueError
at decorator time.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang.ttl_api import (
    pykernel_gen,
    Program,
    CircularBuffer,
    TensorAccessor,
    compute,
    datamovement,
)
from ttlang.operators import copy

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


# CHECK: ValueError: Invalid memory_space: 'INVALID'
# CHECK: Must be one of:
@pykernel_gen(
    grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1)], memory_space="INVALID"
)
def invalid_memory_space_kernel(lhs, rhs, out):
    """This kernel should fail because memory_space='INVALID' is not supported."""
    lhs_accessor = TensorAccessor(lhs)
    rhs_accessor = TensorAccessor(rhs)
    out_accessor = TensorAccessor(out)

    @compute()
    def add_compute(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        l = lhs_cb.wait()
        r = rhs_cb.wait()
        o = out_cb.reserve()
        result = l + r
        o.store(result)
        lhs_cb.pop()
        rhs_cb.pop()
        out_cb.push()

    @datamovement()
    def dm_read(lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer):
        lhs_cb.reserve()
        tx_lhs = copy(lhs_accessor[0, 0], lhs_cb)
        tx_lhs.wait()
        lhs_cb.push()

        rhs_cb.reserve()
        tx_rhs = copy(rhs_accessor[0, 0], rhs_cb)
        tx_rhs.wait()
        rhs_cb.push()

    @datamovement()
    def dm_write(
        lhs_cb: CircularBuffer, rhs_cb: CircularBuffer, out_cb: CircularBuffer
    ):
        out_cb.wait()
        tx = copy(out_cb, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(add_compute, dm_read, dm_write)(lhs, rhs, out)


if __name__ == "__main__":
    # The error should be raised at module load time (decorator execution)
    # This code should never be reached
    print("ERROR: Expected ValueError was not raised at decorator time!")
    exit(1)
