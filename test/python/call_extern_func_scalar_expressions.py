# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.i32.mlir SCALAR_WIDTH=i32 %python %s > %t.i32.cpp 2>&1
# RUN: FileCheck %s --check-prefix=I32 < %t.i32.mlir
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.i64.mlir SCALAR_WIDTH=i64 %python %s > %t.i64.cpp 2>&1
# RUN: FileCheck %s --check-prefix=I64 < %t.i64.mlir

"""Verify external scalar results mix with same-width integer literals."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/scalar_result.hpp"
RESULT_TYPE = {
    "i32": ttl.ScalarType.I32,
    "i64": ttl.ScalarType.I64,
}[os.environ["SCALAR_WIDTH"]]


@ttl.operation(grid=(1, 1))
def scalar_expressions(inp):
    @ttl.compute()
    def compute():
        value = ttl.call_extern_func(
            FAKE_HEADER, "scalar_result", result_type=RESULT_TYPE
        )
        if value == 1:
            ttl.call_extern_func(FAKE_HEADER, "consume_eq", func_args=[value])
        if value != 0:
            ttl.call_extern_func(FAKE_HEADER, "consume_ne", func_args=[value])
        incremented = value + 1
        ttl.call_extern_func(FAKE_HEADER, "consume_add", func_args=[incremented])
        if value < 2:
            ttl.call_extern_func(FAKE_HEADER, "consume_lt", func_args=[value])
        if 1 == value:
            ttl.call_extern_func(FAKE_HEADER, "consume_reverse_eq", func_args=[value])
        reverse_incremented = 1 + value
        ttl.call_extern_func(
            FAKE_HEADER, "consume_reverse_add", func_args=[reverse_incremented]
        )
        if 2 > value:
            ttl.call_extern_func(FAKE_HEADER, "consume_reverse_gt", func_args=[value])
        direct_incremented: int = (
            ttl.call_extern_func(
                FAKE_HEADER, "direct_scalar_result", result_type=RESULT_TYPE
            )
            + 1
        )
        ttl.call_extern_func(
            FAKE_HEADER, "consume_direct_add", func_args=[direct_incremented]
        )
        direct_scaled: int = (
            ttl.call_extern_func(
                FAKE_HEADER, "direct_scalar_product", result_type=RESULT_TYPE
            )
            * 2
        )
        ttl.call_extern_func(
            FAKE_HEADER, "consume_direct_product", func_args=[direct_scaled]
        )

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        pass


# I32: ttl.opaque_call "scalar_result" () {{.*}} : () -> i32
# I32: arith.cmpi eq, {{.*}} : i32
# I32: arith.cmpi ne, {{.*}} : i32
# I32: arith.addi {{.*}} : i32
# I32: arith.cmpi slt, {{.*}} : i32
# I32-COUNT-1: ttl.opaque_call "direct_scalar_result"
# I32-COUNT-1: ttl.opaque_call "direct_scalar_product"
# I32: arith.muli {{.*}} : i32
# I32-NOT: memref.alloca
# I64: ttl.opaque_call "scalar_result" () {{.*}} : () -> i64
# I64: arith.cmpi eq, {{.*}} : i64
# I64: arith.cmpi ne, {{.*}} : i64
# I64: arith.addi {{.*}} : i64
# I64: arith.cmpi slt, {{.*}} : i64
# I64-COUNT-1: ttl.opaque_call "direct_scalar_result"
# I64-COUNT-1: ttl.opaque_call "direct_scalar_product"
# I64: arith.muli {{.*}} : i64
# I64-NOT: memref.alloca


if __name__ == "__main__":
    host_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        host_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    scalar_expressions(input_tensor)
