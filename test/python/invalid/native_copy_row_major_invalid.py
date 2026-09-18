# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""Native tensor copies require tiled storage; external calls accept row-major."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn

import ttl


@ttl.operation(grid=(1, 1))
def native_copy_row_major_invalid(inp):
    transfer_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

    @ttl.compute()
    def compute():
        block = transfer_dfb.wait()
        block.pop()

    @ttl.datamovement()
    def dm_read():
        block = transfer_dfb.reserve()
        transfer = ttl.copy(inp[0, 0], block)
        transfer.wait()
        block.push()

    @ttl.datamovement()
    def dm_write():
        pass


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        inp = ttnn.from_torch(
            torch.arange(32, dtype=torch.float32).reshape(1, 32).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # CHECK: cannot mix tiled and non-tiled element types
        native_copy_row_major_invalid(inp)
    finally:
        ttnn.close_device(device)
