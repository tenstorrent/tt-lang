# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: fp32_dest_acc_en conflicts with compute_config.
"""

import ttnn
import ttl


compute_config = ttnn.ComputeConfigDescriptor()
compute_config.fp32_dest_acc_en = True


# CHECK: ValueError: fp32_dest_acc_en conflicts with compute_config:
@ttl.kernel(grid=(1, 1), fp32_dest_acc_en=False, compute_config=compute_config)
def conflict_kernel_fp32(a, b, out):
    return None


if __name__ == "__main__":
    conflict_kernel_fp32(None, None, None)
