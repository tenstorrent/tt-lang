# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""Compile repeated composition with one callee-owned logical kernel."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/fake_shim.hpp"
reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)


@ttl.operation()
def selected_helper(inp):
    ttl.call_extern_func(FAKE_HEADER, "reader_entry", kernel=reader)


@ttl.operation(grid=(1, 1))
def selected_caller(inp):
    selected_helper(inp)
    selected_helper(inp)
    selected_helper(inp)


# CHECK-LABEL: func.func @selected_caller__ncrisc
# CHECK-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "__main__.selected_helper[captures={{[0-9a-f]+}}]">
# CHECK-COUNT-3: ttl.opaque_call "reader_entry"
# CHECK-NOT: ttl.logical_kernel = #ttl.logical_kernel<{{.*}}_inl_


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    selected_caller(inp)
