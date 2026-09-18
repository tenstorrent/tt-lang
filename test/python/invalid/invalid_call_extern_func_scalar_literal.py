# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not env TTLANG_COMPILE_ONLY=1 %python %s 2>&1 | FileCheck %s

"""Verify integer literals must fit the external scalar result width."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/scalar_result.hpp"


@ttl.operation(grid=(1, 1))
def out_of_range_scalar_literal(inp):
    @ttl.compute()
    def compute():
        narrow = ttl.call_extern_func(
            FAKE_HEADER, "narrow", result_type=ttl.ScalarType.I32
        )
        combined = narrow + 2147483648
        ttl.call_extern_func(FAKE_HEADER, "consume", func_args=[combined])

    @ttl.datamovement()
    def reader():
        pass

    @ttl.datamovement()
    def writer():
        pass


# CHECK: TTLangCompileError: error: integer literal 2147483648 does not fit in signed i32


if __name__ == "__main__":
    input_tensor = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    out_of_range_scalar_literal(input_tensor)
