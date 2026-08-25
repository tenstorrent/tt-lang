# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=INITIAL < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""Verify repeated unsummarized DFB arguments remain opaque and valid."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/opaque_dfb.hpp"


@ttl.operation()
def use_opaque_dfb_twice(descriptor: ttl.DFB):
    ttl.call_extern_func(
        FAKE_HEADER,
        "use_opaque_dfb_twice",
        func_args=[descriptor, descriptor],
        kernel=ttl.KernelKind.COMPUTE,
    )


@ttl.operation(grid=(1, 1))
def repeated_opaque_dfb(input_tensor):
    descriptor = ttl.make_dataflow_buffer_like(
        input_tensor, shape=(1, 1), block_count=1
    )
    use_opaque_dfb_twice(descriptor)


# Repeated automatic dependency positions remain separate opaque occurrences.
# INITIAL-LABEL: func.func @repeated_opaque_dfb__trisc
# INITIAL: %[[DESCRIPTOR:.*]] = ttl.bind_cb
# INITIAL: ttl.opaque_call "use_opaque_dfb_twice" (%[[DESCRIPTOR]], %[[DESCRIPTOR]]) {header = "/dev/null/opaque_dfb.hpp"}

# CHECK-CPP: use_opaque_dfb_twice(get_compile_time_arg_val(0), get_compile_time_arg_val(0));


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    repeated_opaque_dfb(input_tensor)
