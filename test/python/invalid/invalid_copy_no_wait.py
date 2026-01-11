# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# Test 1: Pretty error output (default, no verbose errors)
# RUN: not %python %s 2>&1 | FileCheck %s --check-prefix=PRETTY

# Test 2: Verbose error output includes raw MLIR diagnostic
# RUN: env TTLANG_VERBOSE_ERRORS=1 not %python %s 2>&1 | FileCheck %s --check-prefix=VERBOSE

"""
Validation test: copy operations must be waited on.

This test verifies that forgetting to call tx.wait() after ttl.copy raises
an MLIR verification error with source location information.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

from ttlang import make_circular_buffer_like, ttl
from ttlang.operators import copy
from ttlang.ttl_api import Program

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def copy_no_wait_kernel(lhs, out):
    """Kernel that forgets to wait on a copy - should fail MLIR verification."""
    lhs_cb = make_circular_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    out_cb = make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_thread():
        l = lhs_cb.wait()
        o = out_cb.reserve()
        o.store(l)
        lhs_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        lhs_cb.reserve()
        tx = copy(lhs[0, 0], lhs_cb)
        # BUG: Forgot to call tx.wait() - this should be caught by MLIR verifier
        lhs_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_cb.wait()
        tx = copy(out_cb, out[0, 0])
        tx.wait()
        out_cb.pop()

    return Program(compute_thread, dm_read, dm_write)(lhs, out)


# PRETTY: error: expects transfer handle to be synchronized with ttl.wait
# PRETTY-NEXT:   --> {{.*}}invalid_copy_no_wait.py:50:10
# PRETTY-NEXT:    |
# PRETTY-NEXT: 50 |         tx = copy(lhs[0, 0], lhs_cb)
# PRETTY-NEXT:    |          ^
# PRETTY-NEXT:    |

# VERBOSE: error: expects transfer handle to be synchronized with ttl.wait
# VERBOSE: MLIR diagnostic:
# VERBOSE: 'ttl.copy' op expects transfer handle to be synchronized with ttl.wait


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        lhs = ttnn.to_memory_config(lhs, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        # This should raise an error because copy is not waited on
        copy_no_wait_kernel(lhs, out)

        print("ERROR: Expected verification error was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
