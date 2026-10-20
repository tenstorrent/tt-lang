# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: not %python %s 2>&1 | FileCheck %s

"""Verify diagnostics for invalid external template-argument expansion."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

FAKE_HEADER = "/dev/null/fake_shim.hpp"
NOT_A_SEQUENCE = 17


@ttl.operation(grid=(1, 1))
def invalid_extern_template_expansion(inp):
    @ttl.compute()
    def compute():
        # CHECK: TTLangCompileError: error: ttl.call_extern_func() starred template arguments must reference a captured or module-level list or tuple
        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[*NOT_A_SEQUENCE],
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
    invalid_extern_template_expansion(inp)
