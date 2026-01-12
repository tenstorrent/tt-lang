# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: single-tile CB requires index syntax for tensor slices.

When a CB has shape 1x1, tensor indexing must use index syntax
(e.g., tensor[0, 0]) rather than range syntax (e.g., tensor[0:1, 0:1]).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
from ttlang import ttl


# CHECK: error: CB shape [1, 1] requires index syntax (e.g., tensor[0, 0]), but got range syntax (e.g., tensor[0:2, 0:2])
@ttl.kernel(grid=(1, 1))
def invalid_singletile_range_kernel(inp, out):
    """This kernel should fail: 1x1 CB but using range syntax."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        x = inp_cb.wait()
        o = out_cb.reserve()
        o.store(x)
        inp_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        # INVALID: using range syntax with 1x1 CB
        tx = ttl.copy(inp[0:1, 0:1], inp_blk)
        tx.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_cb.pop()

    return ttl.Program(compute_fn, dm_read, dm_write)(inp, out)


if __name__ == "__main__":
    import torch

    print("=== Single-tile CB Range Syntax Validation Test ===")

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        invalid_singletile_range_kernel(inp, out)

        print("ERROR: Expected ValueError was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
