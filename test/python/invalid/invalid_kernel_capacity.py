# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject logical-kernel requirements beyond the target-provided capacity."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

FAKE_HEADER = "/dev/null/fake_shim.hpp"


# CHECK: operation requires 3 data_movement kernels, but the target supports 2
@ttl.operation(grid=(1, 1))
def invalid_kernel_capacity(inp):
    first = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    second = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    third = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    ttl.call_extern_func(FAKE_HEADER, "first", kernel=first)
    ttl.call_extern_func(FAKE_HEADER, "second", kernel=second)
    ttl.call_extern_func(FAKE_HEADER, "third", kernel=third)


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_kernel_capacity(inp)
