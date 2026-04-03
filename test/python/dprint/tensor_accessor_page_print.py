# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TT_METAL_DPRINT_CORES=0,0 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Test tensor accessor page printing in datamovement kernels.

Prints raw tensor pages via print(tensor, num_pages=N) in dm_read
(before DMA copy) and dm_write (after DMA write). The tensor data
is accessed through TensorAccessor + noc_async_read_tile.
"""

import os

os.environ["TT_METAL_DPRINT_CORES"] = "0,0"

import torch
import ttnn
import ttl


@ttl.operation(grid=(1, 1))
def tensor_accessor_page_print_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as i, out_dfb.reserve() as o:
            result = ttl.exp(i)
            o.store(result)

    @ttl.datamovement()
    def dm_read():
        print("before copy")
        print(inp, num_pages=1)
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


# The input tensor has random bf16 data. The page print should show
# non-zero values. We check for the label and a page line with at
# least some non-zero float-like output (digits with a decimal point).
# CHECK: before copy
# CHECK: 0: {{.*[0-9]+\.[0-9].*}}

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

    tensor_accessor_page_print_kernel(inp, out)

finally:
    ttnn.close_device(device)
