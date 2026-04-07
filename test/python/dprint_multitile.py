# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TT_METAL_DPRINT_CORES=1,1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Multi-tile tile printing test.

Verifies that tile print inside a fused compute with multi-tile
blocks (shape=(2,2)) fires once per tile iteration (4 times for 2x2).
"""

import os

os.environ["TT_METAL_DPRINT_CORES"] = "1,1"

import torch
import ttnn
import ttl


@ttl.operation(grid=(2, 2))
def multitile_dst_kernel(inp, inp2, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(2, 2), block_count=2)
    inp2_dfb = ttl.make_dataflow_buffer_like(inp2, shape=(2, 2), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 2), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as lhs, inp2_dfb.wait() as rhs, out_dfb.reserve() as o:
            exp_lhs = ttl.exp(lhs)
            exp_rhs = ttl.exp(rhs)
            add_result = exp_lhs + exp_rhs
            o.store(add_result)
            print(o, thread="pack")

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0:2, 0:2], blk)
            tx.wait()
        with inp2_dfb.reserve() as blk2:
            tx2 = ttl.copy(inp2[0:2, 0:2], blk2)
            tx2.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:2, 0:2])
            tx.wait()


# =============================================================================
# FileCheck patterns
#
# The fused compute (exp+exp+add) iterates over 2x2=4 tiles.
# The tile print on pack thread fires once per tile iteration.
# Each tile print produces ====== (start) and ++++++ (end) delimiters.
# Output lines are prefixed with "0:(x=1,y=1):TR2:" by the device
# dprint infrastructure (device 0, core 1,1, pack thread = TR2).
# =============================================================================

# First tile iteration
# CHECK: ======
# CHECK: ++++++

# Second tile iteration
# CHECK: ======
# CHECK: ++++++

# Third tile iteration
# CHECK: ======
# CHECK: ++++++

# Fourth tile iteration
# CHECK: ======
# CHECK: ++++++


# =============================================================================
# Test execution
# =============================================================================

device = ttnn.open_device(device_id=0)

try:
    inp_bf16 = ttnn.from_torch(
        torch.randn((128, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    inp2_bf16 = ttnn.from_torch(
        torch.randn((128, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    out_bf16 = ttnn.from_torch(
        torch.zeros((128, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    multitile_dst_kernel(inp_bf16, inp2_bf16, out_bf16)

finally:
    ttnn.close_device(device)
