# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""Verify that compute kernels reject tensor-derived raw addresses."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn


FAKE_HEADER = "/dev/null/fake_shim.hpp"


@ttl.operation(grid=(1, 1))
def invalid_raw_addr_compute(inp):
    @ttl.compute()
    def compute():
        # CHECK: error: requires an enclosing data movement (noc) kernel thread
        ttl.call_extern_func(
            FAKE_HEADER,
            "use_addr",
            func_args=[ttl.raw_addr(inp)],
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        invalid_raw_addr_compute(inp)
    finally:
        ttnn.close_device(device)
