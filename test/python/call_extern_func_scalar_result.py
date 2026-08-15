# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.explicit.mlir SCALAR_RESULT_MODE=explicit %python %s > %t.explicit.cpp 2>&1
# RUN: FileCheck %s --check-prefix=EXPLICIT < %t.explicit.mlir
# RUN: FileCheck %s --check-prefix=EXPLICIT-CPP < %t.explicit.cpp
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.composed.mlir SCALAR_RESULT_MODE=composed %python %s > %t.composed.cpp 2>&1
# RUN: FileCheck %s --check-prefix=COMPOSED < %t.composed.mlir
# RUN: FileCheck %s --check-prefix=COMPOSED-CPP < %t.composed.cpp
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.none.mlir SCALAR_RESULT_MODE=none %python %s > %t.none.cpp 2>&1
# RUN: FileCheck %s --check-prefix=NONE < %t.none.mlir
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.closure-none.mlir SCALAR_RESULT_MODE=closure-none %python %s > %t.closure-none.cpp 2>&1
# RUN: FileCheck %s --check-prefix=CLOSURE-NONE < %t.closure-none.mlir
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.closure-class.mlir SCALAR_RESULT_MODE=closure-class %python %s > %t.closure-class.cpp 2>&1
# RUN: FileCheck %s --check-prefix=CLOSURE-CLASS < %t.closure-class.mlir

"""Verify typed scalar results in explicit and composed operations."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ScalarType


FAKE_HEADER = "/dev/null/scalar_result.hpp"


def make_explicit_scalar_result(result_type):
    @ttl.operation(grid=(1, 1))
    def explicit_scalar_result(inp):
        @ttl.compute()
        def compute():
            predicate: int = ttl.call_extern_func(
                FAKE_HEADER,
                "scalar_result_i32",
                result_type=result_type,
            )
            ttl.call_extern_func(
                FAKE_HEADER,
                "consume_i32",
                func_args=[predicate],
            )

        @ttl.datamovement()
        def reader():
            pass

        @ttl.datamovement()
        def writer():
            pass

    return explicit_scalar_result


explicit_scalar_result = make_explicit_scalar_result(ttl.ScalarType.I32)


def make_explicit_none_result(result_type):
    @ttl.operation(grid=(1, 1))
    def explicit_none_result(inp):
        @ttl.compute()
        def compute():
            ttl.call_extern_func(FAKE_HEADER, "captured_none", result_type=result_type)

        @ttl.datamovement()
        def reader():
            pass

        @ttl.datamovement()
        def writer():
            pass

    return explicit_none_result


explicit_closure_none_result = make_explicit_none_result(None)


def make_explicit_class_result(scalar_types):
    @ttl.operation(grid=(1, 1))
    def explicit_class_result(inp):
        @ttl.compute()
        def compute():
            ttl.call_extern_func(
                FAKE_HEADER,
                "captured_class",
                result_type=scalar_types.I64,
            )

        @ttl.datamovement()
        def reader():
            pass

        @ttl.datamovement()
        def writer():
            pass

    return explicit_class_result


explicit_closure_class_result = make_explicit_class_result(ttl.ScalarType)


@ttl.operation()
def composed_scalar_helper():
    predicate: int = ttl.call_extern_func(
        FAKE_HEADER,
        "scalar_result_i64",
        result_type=ScalarType.I64,
        kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
    )
    if predicate:
        ttl.call_extern_func(
            FAKE_HEADER,
            "consume_i64",
            func_args=[predicate],
            kernel=(ttl.KernelKind.COMPUTE, ttl.KernelKind.DATA_MOVEMENT),
        )


@ttl.operation(grid=(1, 1))
def composed_scalar_result(inp):
    composed_scalar_helper()


VOID_RESULT_TYPE = None


@ttl.operation(grid=(1, 1))
def explicit_none_result(inp):
    @ttl.compute()
    def compute():
        ttl.call_extern_func(FAKE_HEADER, "explicit_none", result_type=None)
        ttl.call_extern_func(FAKE_HEADER, "captured_none", result_type=VOID_RESULT_TYPE)

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        pass


# EXPLICIT-LABEL: func.func @compute
# EXPLICIT: %[[PREDICATE:.*]] = ttl.opaque_call "scalar_result_i32" () {header = "/dev/null/scalar_result.hpp"} : () -> i32
# EXPLICIT-NEXT: ttl.opaque_call "consume_i32" (%[[PREDICATE]]) {header = "/dev/null/scalar_result.hpp"} : (i32) -> ()
# EXPLICIT-NOT: memref.alloca
# EXPLICIT-CPP: scalar_result_i32()
# EXPLICIT-CPP: consume_i32({{.*}})

# COMPOSED-LABEL: func.func @composed_scalar_result__trisc
# COMPOSED: %[[COMPUTE_PREDICATE:.*]] = ttl.opaque_call "scalar_result_i64" () {header = "/dev/null/scalar_result.hpp"} : () -> i64
# COMPOSED: ttl.opaque_call "consume_i64" (%[[COMPUTE_PREDICATE]]) {header = "/dev/null/scalar_result.hpp"} : (i64) -> ()
# COMPOSED-LABEL: func.func @composed_scalar_result__ncrisc
# COMPOSED: %[[DATA_MOVEMENT_PREDICATE:.*]] = ttl.opaque_call "scalar_result_i64" () {header = "/dev/null/scalar_result.hpp"} : () -> i64
# COMPOSED: ttl.opaque_call "consume_i64" (%[[DATA_MOVEMENT_PREDICATE]]) {header = "/dev/null/scalar_result.hpp"} : (i64) -> ()
# COMPOSED-NOT: memref.alloca
# COMPOSED-CPP: scalar_result_i64()
# COMPOSED-CPP: consume_i64({{.*}})

# NONE-LABEL: func.func @compute
# NONE: ttl.opaque_call "explicit_none" () {header = "/dev/null/scalar_result.hpp"} : () -> ()
# NONE: ttl.opaque_call "captured_none" () {header = "/dev/null/scalar_result.hpp"} : () -> ()
# CLOSURE-NONE-LABEL: func.func @compute
# CLOSURE-NONE: ttl.opaque_call "captured_none" () {header = "/dev/null/scalar_result.hpp"} : () -> ()
# CLOSURE-CLASS-LABEL: func.func @compute
# CLOSURE-CLASS: ttl.opaque_call "captured_class" () {header = "/dev/null/scalar_result.hpp"} : () -> i64


if __name__ == "__main__":
    host_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        host_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    operation = {
        "explicit": explicit_scalar_result,
        "composed": composed_scalar_result,
        "none": explicit_none_result,
        "closure-none": explicit_closure_none_result,
        "closure-class": explicit_closure_class_result,
    }[os.environ["SCALAR_RESULT_MODE"]]
    operation(input_tensor)
