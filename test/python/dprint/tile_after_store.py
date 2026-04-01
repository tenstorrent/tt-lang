# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TT_METAL_DPRINT_CORES=0,0 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Test tile print before and after store in compute.

Before store, the output CB is empty so tile print shows zeros.
After store, the CB contains exp(input) data so tile print shows
non-zero values.
"""

import os

os.environ["TT_METAL_DPRINT_CORES"] = "0,0"

import torch
import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def tile_after_store_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            result = ttl.exp(i)
            print("before store", thread="pack")
            print(o)
            o.store(result)
            print("after store", thread="pack")
            print(o)

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# Before store: output CB is uninitialized (zeros on sim, garbage on HW).
# CHECK: before store
# CHECK: ======
# CHECK: 0 :
# CHECK: ++++++

# After store: CB now has exp(random_input), expect non-zero floats.
# CHECK: after store
# CHECK: ======
# CHECK: 0 : {{.*[0-9]+\.[0-9].*}}
# CHECK: ++++++

device = ttnn.open_device(device_id=0)

try:
    inp = ttnn.from_torch(
        torch.randn((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tile_after_store_kernel(inp, out)

finally:
    ttnn.close_device(device)
