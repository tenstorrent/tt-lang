# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Test: acc=True loop accumulation pattern
import os
import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def acc_loop_kernel(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), buffer_factor=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with lhs_dfb.wait() as l, rhs_dfb.wait() as r:
            with out_dfb.reserve() as o:
                o.store(l)
                for i in range(4):
                    o.store(r, acc=True)

    @ttl.datamovement()
    def dm_read():
        with lhs_dfb.reserve() as blk:
            tx = ttl.copy(lhs[0, 0], blk)
            tx.wait()
        with rhs_dfb.reserve() as blk:
            tx = ttl.copy(rhs[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)
    try:
        to_dev = lambda v: ttnn.to_memory_config(
            ttnn.from_torch(
                torch.full((32, 32), v, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        acc_loop_kernel(to_dev(2.0), to_dev(3.0), to_dev(0.0))
        print("Compiled successfully")
    finally:
        ttnn.close_device(device)
