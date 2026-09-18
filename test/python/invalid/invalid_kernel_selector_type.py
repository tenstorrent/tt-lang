# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject an external-call selector that is not a logical kernel value."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

FAKE_HEADER = "/dev/null/fake_shim.hpp"


# CHECK: kernel selector must be a KernelKind or Kernel declared as a top-level operation resource
@ttl.operation(grid=(1, 1))
def invalid_kernel_type(inp):
    ttl.call_extern_func(FAKE_HEADER, "external_entry", kernel=42)


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_kernel_type(inp)
