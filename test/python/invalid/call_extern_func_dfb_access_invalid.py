# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s not_list 2>&1 | FileCheck %s --check-prefix=NOT-LIST
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s wrong_access 2>&1 | FileCheck %s --check-prefix=WRONG-ACCESS
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s arguments 2>&1 | FileCheck %s --check-prefix=ARGUMENTS
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s missing_dependency 2>&1 | FileCheck %s --check-prefix=MISSING-DEPENDENCY
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s summarized_repeated_automatic 2>&1 | FileCheck %s --check-prefix=SUMMARIZED-REPEATED-AUTOMATIC
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s summarized_repeated_mixed 2>&1 | FileCheck %s --check-prefix=SUMMARIZED-REPEATED-MIXED
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s aliased_automatic 2>&1 | FileCheck %s --check-prefix=ALIASED-AUTOMATIC

"""Verify diagnostics for malformed external DFB access summaries."""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/inspect_dfb.hpp"
MODE = sys.argv[1]


def make_invalid_dfb_access(mode):
    if mode == "not_list":

        @ttl.operation(grid=(1, 1))
        def invalid_dfb_access(input_tensor):
            descriptor = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=1
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "inspect_dfb",
                template_args=[ttl.dfb_descriptor(descriptor)],
                dfb_accesses=(ttl.DFBAccess.inspect(descriptor),),
                kernel=ttl.KernelKind.COMPUTE,
            )

    elif mode == "wrong_access":

        @ttl.operation(grid=(1, 1))
        def invalid_dfb_access(input_tensor):
            descriptor = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=1
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "inspect_dfb",
                template_args=[ttl.dfb_descriptor(descriptor)],
                dfb_accesses=[ttl.DFBEffect.wait(descriptor, tiles=1)],
                kernel=ttl.KernelKind.COMPUTE,
            )

    elif mode == "arguments":

        @ttl.operation(grid=(1, 1))
        def invalid_dfb_access(input_tensor):
            descriptor = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=1
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "inspect_dfb",
                template_args=[ttl.dfb_descriptor(descriptor)],
                dfb_accesses=[ttl.DFBAccess.inspect(descriptor, descriptor)],
                kernel=ttl.KernelKind.COMPUTE,
            )

    elif mode == "missing_dependency":

        @ttl.operation(grid=(1, 1))
        def invalid_dfb_access(input_tensor):
            descriptor = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=1
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "inspect_dfb",
                dfb_accesses=[ttl.DFBAccess.inspect(descriptor)],
                kernel=ttl.KernelKind.COMPUTE,
            )

    elif mode == "summarized_repeated_automatic":

        @ttl.operation()
        def repeat_descriptor(descriptor: ttl.DFB):
            ttl.call_extern_func(
                FAKE_HEADER,
                "inspect_dfb_twice",
                func_args=[descriptor, descriptor],
                dfb_accesses=[
                    ttl.DFBAccess.inspect(descriptor),
                    ttl.DFBAccess.inspect(descriptor),
                ],
                kernel=ttl.KernelKind.COMPUTE,
            )

        @ttl.operation(grid=(1, 1))
        def invalid_dfb_access(input_tensor):
            descriptor = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=1
            )
            repeat_descriptor(descriptor)

    elif mode == "summarized_repeated_mixed":

        @ttl.operation()
        def repeat_descriptor(descriptor: ttl.DFB):
            ttl.call_extern_func(
                FAKE_HEADER,
                "inspect_dfb_twice",
                template_args=[ttl.dfb_descriptor(descriptor)],
                func_args=[descriptor],
                dfb_accesses=[
                    ttl.DFBAccess.inspect(descriptor),
                    ttl.DFBAccess.inspect(descriptor),
                ],
                kernel=ttl.KernelKind.COMPUTE,
            )

        @ttl.operation(grid=(1, 1))
        def invalid_dfb_access(input_tensor):
            descriptor = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=1
            )
            repeat_descriptor(descriptor)

    elif mode == "aliased_automatic":

        @ttl.operation()
        def repeat_descriptor(descriptor: ttl.DFB):
            alias = descriptor
            ttl.call_extern_func(
                FAKE_HEADER,
                "inspect_dfb_twice",
                func_args=[descriptor, alias],
                dfb_accesses=[
                    ttl.DFBAccess.inspect(descriptor),
                    ttl.DFBAccess.inspect(alias),
                ],
                kernel=ttl.KernelKind.COMPUTE,
            )

        @ttl.operation(grid=(1, 1))
        def invalid_dfb_access(input_tensor):
            descriptor = ttl.make_dataflow_buffer_like(
                input_tensor, shape=(1, 1), block_count=1
            )
            repeat_descriptor(descriptor)

    else:
        raise ValueError(f"unsupported mode: {mode}")

    return invalid_dfb_access


invalid_dfb_access = make_invalid_dfb_access(MODE)


# NOT-LIST: TTLangCompileError: error: ttl.call_extern_func() dfb_accesses must be a list
# WRONG-ACCESS: TTLangCompileError: error: ttl.call_extern_func() dfb_accesses element must be ttl.DFBAccess.inspect
# ARGUMENTS: TTLangCompileError: error: ttl.DFBAccess.inspect() requires exactly one DFB argument
# MISSING-DEPENDENCY: TTLangCompileError: error: ttl.call_extern_func() DFB access references a DFB that is not a function argument, descriptor, or dependency
# SUMMARIZED-REPEATED-AUTOMATIC: TTLangCompileError: error: ttl.call_extern_func() DFB access reference is ambiguous because the DFB appears in multiple dependency positions; use a distinct composed-operation DFB parameter for each position and reference it directly
# SUMMARIZED-REPEATED-MIXED: TTLangCompileError: error: ttl.call_extern_func() DFB access reference is ambiguous because the DFB appears in multiple dependency positions; use a distinct composed-operation DFB parameter for each position and reference it directly
# ALIASED-AUTOMATIC: TTLangCompileError: error: ttl.call_extern_func() DFB access reference is ambiguous because the DFB appears in multiple dependency positions; use a distinct composed-operation DFB parameter for each position and reference it directly


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_dfb_access(input_tensor)
