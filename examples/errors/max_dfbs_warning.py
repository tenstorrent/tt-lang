# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# TTLANG_HARDWARE_CI: skip-compiler
"""Test: how many CBs can we allocate?"""

import torch
import numpy as np
import ttnn
import ttl

TILE = 32


@ttl.operation(grid=(1, 1))
def test_many_cbs(inp, out):
    cb0 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb1 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb2 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb3 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb4 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb5 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb6 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb7 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb8 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb9 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb10 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb11 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb12 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb13 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb14 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb15 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb16 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb17 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb18 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb19 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb20 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb21 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb22 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb23 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb24 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb25 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb26 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb27 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb28 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb29 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb30 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb31 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb32 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb33 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb34 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    cb35 = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        # Just use a few to make sure it compiles
        with cb0.wait() as a, cb35.reserve() as o:
            o.store(a + a)
        with cb35.wait() as a, cb1.reserve() as o:
            o.store(a)

    @ttl.datamovement()
    def dm_read():
        with cb0.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with cb1.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
    out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

    inp_tt = ttnn.from_torch(
        inp_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.from_torch(
        out_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    test_many_cbs(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt).float()
    expected = inp_torch.float() * 2.0

    print(f"Result[0,0]: {result[0,0]:.4f}, Expected: {expected[0,0]:.4f}")
    err = torch.max(torch.abs(result - expected) / (torch.abs(expected) + 1e-6))
    print(f"Max rel error: {err:.4f}")
    print(f"36 CBs: {'PASS' if err < 0.05 else 'FAIL'}")

    ttnn.close_device(device)
