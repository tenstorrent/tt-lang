# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: broadcast only supports 2D tensors.

Broadcast is a hardware-level 2D operation. ND tensors must use
elementwise ops instead.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


# CHECK: broadcast only supports 2D tensors, got rank 3
@ttl.kernel(grid=(1, 1))
def invalid_nd_bcast_kernel(inp, out):
    """This kernel should fail: bcast on 3D tensor."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as x, out_dfb.reserve() as o:
            result = ttl.math.broadcast(x, o, dims=[0])
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0, 0, 0], inp_blk)
        tx.wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0, 0])
        tx.wait()
        out_blk.pop()


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.full((1, 32, 32), 1.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((1, 32, 32), dtype=torch.bfloat16)

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

        invalid_nd_bcast_kernel(inp, out)

        print("ERROR: Expected ValueError was not raised!")
        exit(1)

    finally:
        ttnn.close_device(device)
