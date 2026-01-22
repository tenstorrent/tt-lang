# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: dst_full_sync_en requires buffer_factor=1.
"""

import ttl


# CHECK: ValueError: dst_full_sync_en=True requires buffer_factor=1, got buffer_factor=2
# CHECK-NEXT:   --> {{.*}}invalid_dst_full_sync_buffer_factor.py:[[LINE:[0-9]+]]:1
# CHECK-NEXT:    |
# CHECK-NEXT: [[LINE]] | @ttl.kernel(grid=(1, 1), dst_full_sync_en=True)
# CHECK-NEXT:    | ^
# CHECK-NEXT:    |
class DummyTensor:
    dtype = "bf16"


@ttl.kernel(grid=(1, 1), dst_full_sync_en=True)
def invalid_dst_full_sync_kernel():
    dummy = DummyTensor()
    cb = ttl.make_circular_buffer_like(dummy, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def add_compute():
        _ = cb


if __name__ == "__main__":
    invalid_dst_full_sync_kernel()
