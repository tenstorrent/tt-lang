# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""Verify that ttl.dfb_descriptor rejects non-DFB values."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/fake_shim.hpp"


@ttl.operation(grid=(1, 1))
def invalid_tensor_descriptor(inp):
    @ttl.compute()
    def compute():
        # CHECK: TTLangCompileError: error: ttl.dfb_descriptor() argument must be a DFB
        ttl.call_extern_func(
            FAKE_HEADER,
            "my_shim",
            template_args=[ttl.dfb_descriptor(inp)],
        )

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        host = torch.ones((32, 32), dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        invalid_tensor_descriptor(inp)
    finally:
        ttnn.close_device(device)
