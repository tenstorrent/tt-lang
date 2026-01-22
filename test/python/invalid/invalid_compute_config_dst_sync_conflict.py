# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: dst_full_sync_en conflicts with compute_config.
"""

import ttnn
import ttl


compute_config = ttnn.ComputeConfigDescriptor()
compute_config.dst_full_sync_en = True


# CHECK: ValueError: dst_full_sync_en conflicts with compute_config:
@ttl.kernel(grid=(1, 1), dst_full_sync_en=False, compute_config=compute_config)
def conflict_kernel_dst_sync(a, b, out):
    return None


if __name__ == "__main__":
    conflict_kernel_dst_sync(None, None, None)
