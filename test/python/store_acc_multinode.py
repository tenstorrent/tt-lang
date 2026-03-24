# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Multinode multi-tile accumulation with two consecutive acc=True stores.
Verifies that the first store packs normally and the second store uses
L1 accumulation (llk_pack_reconfig_l1_acc).
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
GRANULARITY = 2


@ttl.kernel(grid=(2, 2))
def acc_multitile_kernel(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor):
    row_tiles = a.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(GRANULARITY, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(GRANULARITY, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(GRANULARITY, 1), buffer_factor=2
    )

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with (
                            a_dfb.wait() as av,
                            b_dfb.wait() as bv,
                            out_dfb.reserve() as o,
                        ):
                            o.store(av, acc=True)
                            o.store(bv, acc=True)

    @ttl.datamovement()
    def dm_read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            tx_a = ttl.copy(a[r0:r1, col : col + 1], a_blk)
                            tx_b = ttl.copy(b[r0:r1, col : col + 1], b_blk)
                            tx_a.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def dm_write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()


import torch

dim = 256
a = torch.randn(dim, dim, dtype=torch.bfloat16)
b = torch.randn(dim, dim, dtype=torch.bfloat16)
out = torch.zeros(dim, dim, dtype=torch.bfloat16)
acc_multitile_kernel(
    ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT),
    ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT),
    ttnn.from_torch(out, layout=ttnn.TILE_LAYOUT),
)

# First compute (overwrite): no llk_pack_reconfig_l1_acc.
# CHECK: // compute
# CHECK: tile_regs_acquire();
# CHECK: copy_tile
# CHECK: tile_regs_commit();
# CHECK: tile_regs_wait();
# CHECK-NOT: llk_pack_reconfig_l1_acc
# CHECK: pack_tile<true>(
# CHECK-NOT: llk_pack_reconfig_l1_acc
# CHECK: pack_tile<true>(
# CHECK: tile_regs_release();

# Second compute (L1 accumulation): llk_pack_reconfig_l1_acc wraps pack_tile.
# CHECK: tile_regs_acquire();
# CHECK: copy_tile
# CHECK: tile_regs_commit();
# CHECK: tile_regs_wait();
# CHECK: llk_pack_reconfig_l1_acc(
# CHECK-NEXT: pack_tile<true>(
# CHECK-NEXT: llk_pack_reconfig_l1_acc(
# CHECK: llk_pack_reconfig_l1_acc(
# CHECK-NEXT: pack_tile<true>(
# CHECK-NEXT: llk_pack_reconfig_l1_acc(
# CHECK: tile_regs_release();
