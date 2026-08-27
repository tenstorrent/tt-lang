# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s 2>&1 | FileCheck %s

"""Verify mixed nonconstant scalar widths produce a user diagnostic."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/scalar_result.hpp"


@ttl.operation(grid=(1, 1))
def mixed_scalar_widths(inp):
    @ttl.compute()
    def compute():
        narrow = ttl.call_extern_func(
            FAKE_HEADER, "narrow", result_type=ttl.ScalarType.I32
        )
        wide = ttl.call_extern_func(FAKE_HEADER, "wide", result_type=ttl.ScalarType.I64)
        combined = narrow + wide
        ttl.call_extern_func(FAKE_HEADER, "consume", func_args=[combined])

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        pass


# CHECK: TTLangCompileError: error: integer operands require matching widths, got i32 and i64


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    mixed_scalar_widths(input_tensor)
