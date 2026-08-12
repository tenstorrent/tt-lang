# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.mlir %python %s
# RUN: FileCheck %s < %t.mlir

"""Verify dispatch-condition identity through composition and splitting."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/dispatch_condition.hpp"


def make_dispatch_condition_operation():
    active = ttl.DispatchCondition(ttl.ScalarType.I64)

    @ttl.operation()
    def conditional_helper():
        predicate: int = ttl.call_extern_func(
            FAKE_HEADER,
            "active",
            condition_result=active,
            kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
        )
        if predicate:
            ttl.call_extern_func(
                FAKE_HEADER,
                "consume",
                func_args=[predicate],
                kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
            )

    @ttl.operation(grid=(1, 1))
    def dispatch_condition_operation(input_tensor):
        conditional_helper()

    return dispatch_condition_operation


dispatch_condition_operation = make_dispatch_condition_operation()


# CHECK-LABEL: func.func @dispatch_condition_operation__trisc
# CHECK: %[[COMPUTE:.*]] = ttl.opaque_call "active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "/dev/null/dispatch_condition.hpp"} : () -> i64
# CHECK: scf.if
# CHECK: ttl.opaque_call "consume" (%[[COMPUTE]])
# CHECK-LABEL: func.func @dispatch_condition_operation__ncrisc
# CHECK: %[[DATA_MOVEMENT:.*]] = ttl.opaque_call "active" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "/dev/null/dispatch_condition.hpp"} : () -> i64
# CHECK: scf.if
# CHECK: ttl.opaque_call "consume" (%[[DATA_MOVEMENT]])


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    dispatch_condition_operation(input_tensor)
