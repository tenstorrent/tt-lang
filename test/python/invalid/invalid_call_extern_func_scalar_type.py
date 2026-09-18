# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s string 2>&1 | FileCheck %s --check-prefix=STRING
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s integer 2>&1 | FileCheck %s --check-prefix=INTEGER
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s enum-class 2>&1 | FileCheck %s --check-prefix=ENUM-CLASS

"""Verify that external scalar results require a ScalarType member."""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/scalar_result.hpp"
INVALID_RESULT_TYPES = {
    "string": "i64",
    "integer": 64,
    "enum-class": ttl.ScalarType,
}
INVALID_RESULT_TYPE = INVALID_RESULT_TYPES[sys.argv[1]]


@ttl.operation(grid=(1, 1))
def invalid_scalar_result(inp):
    @ttl.compute()
    def compute():
        ttl.call_extern_func(
            FAKE_HEADER,
            "scalar_result",
            result_type=INVALID_RESULT_TYPE,
        )

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        pass


# STRING: TTLangCompileError: error: ttl.call_extern_func() result_type must be ttl.ScalarType.I32 or ttl.ScalarType.I64, got str
# INTEGER: TTLangCompileError: error: ttl.call_extern_func() result_type must be ttl.ScalarType.I32 or ttl.ScalarType.I64, got int
# ENUM-CLASS: TTLangCompileError: error: ttl.call_extern_func() result_type must be ttl.ScalarType.I32 or ttl.ScalarType.I64, got EnumType


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    invalid_scalar_result(input_tensor)
