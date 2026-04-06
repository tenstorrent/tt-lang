# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: chaining a method call on a void expression must error.

store() returns no value, so chaining .push() on its result is invalid.
A user might write this expecting store to return the block for further
lifecycle operations.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: cannot call .push() on '{{.*}}store{{.*}}': expression does not produce a value
@ttl.operation(grid=(1, 1))
def invalid_chain_kernel(a, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as a_blk, out_dfb.reserve() as o:
            # INVALID: store() returns no value, cannot chain .push()
            o.store(a_blk).push()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_l1

    device = ttnn.open_device(device_id=0)

    try:
        a = to_l1(torch.randn(32, 32, dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

        invalid_chain_kernel(a, out)

        print("ERROR: Expected error was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
