# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Multinode column accumulation: each node accumulates column blocks into a
row-output DFB. Models the inner loop of a row-wise reduction where
multiple column blocks are summed into a single output column per row.

Pattern per node iteration:
    out = col_block[0]                       # overwrite first block
    for k in range(1, cols_per_node):
        out += col_block[k]                  # accumulate remaining blocks
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

TILE_SIZE = 32
DFB_ROWS = 2
NUM_COL_BLOCKS = 3


@ttl.kernel(grid=(2, 2))
def col_accumulate_kernel(inp: ttnn.Tensor, out: ttnn.Tensor):
    total_rows = inp.shape[0] // TILE_SIZE // DFB_ROWS
    total_cols = inp.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-total_rows // grid_rows)

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(DFB_ROWS, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(DFB_ROWS, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        node_x, node_y = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_y * rows_per_node + local_row
            if row < total_rows:
                with out_dfb.reserve() as o:
                    # First column block: overwrite.
                    with inp_dfb.wait() as blk:
                        o.store(blk)
                    # Remaining column blocks: accumulate.
                    for k in range(NUM_COL_BLOCKS - 1):
                        with inp_dfb.wait() as blk:
                            o.store(blk, acc=True)

    @ttl.datamovement()
    def dm_read():
        node_x, node_y = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_y * rows_per_node + local_row
            if row < total_rows:
                r0, r1 = row * DFB_ROWS, (row + 1) * DFB_ROWS
                for col in range(NUM_COL_BLOCKS):
                    with inp_dfb.reserve() as blk:
                        tx = ttl.copy(inp[r0:r1, col : col + 1], blk)
                        tx.wait()

    @ttl.datamovement()
    def dm_write():
        node_x, node_y = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_y * rows_per_node + local_row
            if row < total_rows:
                r0, r1 = row * DFB_ROWS, (row + 1) * DFB_ROWS
                with out_dfb.wait() as blk:
                    tx = ttl.copy(blk, out[r0:r1, 0:1])
                    tx.wait()


import torch

rows = 256
cols = NUM_COL_BLOCKS * TILE_SIZE
inp = torch.randn(rows, cols, dtype=torch.bfloat16)
out = torch.zeros(rows, TILE_SIZE, dtype=torch.bfloat16)
col_accumulate_kernel(
    ttnn.from_torch(inp, layout=ttnn.TILE_LAYOUT),
    ttnn.from_torch(out, layout=ttnn.TILE_LAYOUT),
)

# The accumulating stores use L1 accumulation (pack_reconfig_l1_acc).
# CHECK: // compute
# CHECK: pack_tile<true>(
# CHECK: llk_pack_reconfig_l1_acc(
# CHECK-NEXT: pack_tile<true>(
# CHECK-NEXT: llk_pack_reconfig_l1_acc(
