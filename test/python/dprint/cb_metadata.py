# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TT_METAL_DPRINT_CORES=0,0 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Test CB metadata printing.

Two consecutive print(dfb) calls should produce output on separate
lines, each showing cb_id with size, limit, page_size, etc.
"""

import os

os.environ["TT_METAL_DPRINT_CORES"] = "0,0"

import torch
import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def cb_metadata_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # Two consecutive CB prints should appear on separate lines.
        print(inp_dfb)
        print(out_dfb)
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            o.store(i)

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


# Each CB print should be on its own line with full metadata.
# CHECK: cb_id {{[0-9]+}}: { size:
# CHECK: cb_id {{[0-9]+}}: { size:

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

    cb_metadata_kernel(inp, out)

finally:
    ttnn.close_device(device)
