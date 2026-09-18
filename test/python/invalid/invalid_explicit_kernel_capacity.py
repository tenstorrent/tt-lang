# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject explicit logical threads beyond the target-provided capacity."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


first_compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
second_compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)


# CHECK: operation requires 2 compute kernels, but the target supports 1; selected kernels: compute kernel 'first_compute_kernel', compute kernel 'second_compute_kernel'
@ttl.operation(grid=(1, 1))
def invalid_explicit_kernel_capacity(inp):
    @ttl.compute(kernel=first_compute_kernel)
    def first_compute():
        pass

    @ttl.compute(kernel=second_compute_kernel)
    def second_compute():
        pass

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        pass


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_explicit_kernel_capacity(inp)
