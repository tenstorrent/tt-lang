# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=IDENTITY < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CALLS < %t.initial.mlir
# RUN: env PYTHONHASHSEED=0 TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.seed0.mlir TTLANG_COMPILER_OPTIONS=--ttl-specialize-cores %python %s > %t.seed0.output 2>&1
# RUN: FileCheck %s --check-prefix=SPECIALIZED < %t.seed0.mlir
# RUN: env PYTHONHASHSEED=12345 TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.seed12345.mlir TTLANG_COMPILER_OPTIONS=--ttl-specialize-cores %python %s > %t.seed12345.output 2>&1
# RUN: FileCheck %s --check-prefix=SPECIALIZED < %t.seed12345.mlir

"""Compile distinct factory instances without merging their logical kernels."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/fake_shim.hpp"


def make_helper(entry):
    reader = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)

    @ttl.operation()
    def selected_helper(inp):
        ttl.call_extern_func(FAKE_HEADER, entry, kernel=reader)

    return selected_helper


first_helper = make_helper("first_entry")
second_helper = make_helper("second_entry")


@ttl.operation(grid=(1, 1))
def selected_caller(inp):
    first_helper(inp)
    first_helper(inp)
    second_helper(inp)


# IDENTITY-COUNT-2: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "__main__.make_helper.<locals>.selected_helper[captures={{[0-9a-f]+}}]">
# IDENTITY-NOT: ttl.logical_kernel = #ttl.logical_kernel<{{.*}}_inl_

# CALLS-DAG: ttl.opaque_call "first_entry"
# CALLS-DAG: ttl.opaque_call "first_entry"
# CALLS-DAG: ttl.opaque_call "second_entry"

# SPECIALIZED-LABEL: func.func @selected_caller__ncrisc()
# SPECIALIZED-SAME: ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "__main__.make_helper.<locals>.selected_helper[captures=6142f2ea4f753c85]">
# SPECIALIZED-NEXT: emitc.call_opaque "second_entry"


if __name__ == "__main__":
    inp = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    selected_caller(inp)
