# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: ttl.call_extern_func rejects tensor slice/view extern args.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


FAKE_HEADER = "/dev/null/fake_shim.hpp"


# CHECK: TTLangCompileError: error: ttl.call_extern_func() does not support tensor slices/views in extern arguments yet; pass the base tensor or ttl.raw_addr(base_tensor)
@ttl.operation(grid=(1, 1))
def invalid_slice_arg(inp):
    @ttl.compute()
    def compute():
        ttl.call_extern_func(FAKE_HEADER, "my_shim", func_args=[inp[0, 0]])

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        invalid_slice_arg(inp)
    finally:
        ttnn.close_device(device)
