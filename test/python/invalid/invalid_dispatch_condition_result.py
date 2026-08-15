# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s declaration 2>&1 | FileCheck %s --check-prefix=DECLARATION
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s raw 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s combined 2>&1 | FileCheck %s --check-prefix=COMBINED
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s stateful 2>&1 | FileCheck %s --check-prefix=STATEFUL
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s index 2>&1 | FileCheck %s --check-prefix=INDEX
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s global 2>&1 | FileCheck %s --check-prefix=GLOBAL

"""Verify invalid dispatch-condition declarations and uses."""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/dispatch_condition.hpp"
MODE = sys.argv[1]
GLOBAL_CONDITION = ttl.DispatchCondition(ttl.ScalarType.I32)

if MODE == "declaration":
    ttl.DispatchCondition("i64")


def make_invalid_operation(mode):
    local_condition = ttl.DispatchCondition(ttl.ScalarType.I32)

    if mode == "raw":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            @ttl.compute()
            def compute():
                ttl.call_extern_func(FAKE_HEADER, "raw", condition_result=17)

            @ttl.datamovement()
            def reader():
                pass

            @ttl.datamovement()
            def writer():
                pass

    elif mode == "combined":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            @ttl.compute()
            def compute():
                ttl.call_extern_func(
                    FAKE_HEADER,
                    "combined",
                    result_type=ttl.ScalarType.I32,
                    condition_result=local_condition,
                )

            @ttl.datamovement()
            def reader():
                pass

            @ttl.datamovement()
            def writer():
                pass

    elif mode == "stateful":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            scratch = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                ttl.call_extern_func(
                    FAKE_HEADER,
                    "stateful",
                    dfb_dependencies=[scratch],
                    condition_result=local_condition,
                )

            @ttl.datamovement()
            def reader():
                pass

            @ttl.datamovement()
            def writer():
                pass

    elif mode == "index":

        @ttl.operation(grid=(1, 1))
        def invalid_operation(input_tensor):
            scratch = ttl.make_dfb("bf16", shape=(1, 1), block_count=2)

            @ttl.compute()
            def compute():
                ttl.call_extern_func(
                    FAKE_HEADER,
                    "index",
                    template_args=[ttl.get_dfb_id(scratch)],
                    condition_result=local_condition,
                )

            @ttl.datamovement()
            def reader():
                pass

            @ttl.datamovement()
            def writer():
                pass

    else:
        raise ValueError(f"unsupported test mode: {mode}")

    return invalid_operation


if MODE == "global":

    @ttl.operation(grid=(1, 1))
    def invalid_operation(input_tensor):
        ttl.call_extern_func(
            FAKE_HEADER,
            "global_condition",
            condition_result=GLOBAL_CONDITION,
            kernel=ttl.KernelKind.COMPUTE,
        )

else:
    invalid_operation = make_invalid_operation(MODE)


# DECLARATION: TypeError: DispatchCondition scalar type must be ttl.ScalarType.I32 or ttl.ScalarType.I64, got str
# RAW: TTLangCompileError: error: ttl.call_extern_func() condition_result must be a ttl.DispatchCondition, got int
# COMBINED: TTLangCompileError: error: ttl.call_extern_func() cannot combine result_type and condition_result
# STATEFUL: TTLangCompileError: error: ttl.call_extern_func() condition_result call cannot access DFB state
# INDEX: TTLangCompileError: error: ttl.call_extern_func() condition_result call cannot access DFB state
# GLOBAL: ValueError: @ttl.operation 'invalid_operation': DispatchCondition 'GLOBAL_CONDITION' must be created by an enclosing factory


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_operation(input_tensor)
