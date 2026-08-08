# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""Verify that external float template arguments fit binary32."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/fake_shim.hpp"


@ttl.operation(grid=(1, 1))
def invalid_float_template_argument(inp):
    @ttl.compute()
    def compute():
        # CHECK: TTLangCompileError: error: ttl.call_extern_func() float template argument must be representable as binary32
        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[1e100],
        )

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


if __name__ == "__main__":
    host = torch.ones((32, 32), dtype=torch.bfloat16)
    inp = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    invalid_float_template_argument(inp)
