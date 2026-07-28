# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_CB_TABLE=%t.cb_table.txt %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.cb_table.txt

"""
Test that every compile appends its CB table to TTLANG_CB_TABLE.

The table maps the logical DFB names a kernel was written with to the final
physical CB ids, which the compiler is free to renumber and merge.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.operation(grid=(1, 1))
def cb_table_kernel(lhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=3)

    @ttl.compute()
    def compute_thread():
        l = lhs_dfb.wait()
        o = out_dfb.reserve()
        o.store(l)
        l.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        lhs_blk = lhs_dfb.reserve()
        tx = ttl.copy(lhs[0, 0], lhs_blk)
        tx.wait()
        lhs_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0])
        tx.wait()
        out_blk.pop()


# One stamped block per compile, carrying enough identity to spot a stale file.
# CHECK: === {{[0-9]+}}-{{[0-9]+}}-{{[0-9]+}}T{{[0-9:]+}}Z pid {{[0-9]+}} program_hash {{.*}} source {{.*}}cb_table_dump.py ===
# CHECK: tt-lang CB table: {{[0-9]+}} CBs, {{[0-9]+}} bytes of L1 backing store
# CHECK: id{{ +}}names{{ +}}shape{{ +}}tile{{ +}}blk{{ +}}dtype{{ +}}page{{ +}}pages{{ +}}bytes

# Both logical names resolve, with their declared geometry: one bfloat16 tile
# per block at block_count 2 and 3, so 2 and 3 pages of 2048 bytes.
# CHECK-DAG: {{[0-9]+}}{{ +}}lhs_dfb{{ +}}1x1{{ +}}32x32{{ +}}2{{ +}}BFLOAT16{{ +}}2048{{ +}}2{{ +}}4096
# CHECK-DAG: {{[0-9]+}}{{ +}}out_dfb{{ +}}1x1{{ +}}32x32{{ +}}3{{ +}}BFLOAT16{{ +}}2048{{ +}}3{{ +}}6144


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        lhs_torch = torch.full((32, 32), 2.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        lhs = ttnn.from_torch(
            lhs_torch,
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

        cb_table_kernel(lhs, out)

    finally:
        ttnn.close_device(device)
