# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_FINAL_MLIR=%t.final.mlir %python %s > %t.out 2>&1
# RUN: FileCheck %s < %t.final.mlir --implicit-check-not='emitc.if' --implicit-check-not='scf.if'
# RUN: FileCheck %s --check-prefix=RUNTIME < %t.out

"""Per-core specialization folds the coordinate branch in the real emitted kernel.

The reader swaps two tile-columns based on core_x, so it is cloned per launch
coordinate and each clone's branch is const-folded. This test both executes the
specialized kernel (correctness vs a torch column-swap reference) and FileChecks
the emitted MLIR to show the fold and the dead-read elimination. The MLIR-only
pass check on toy IR lives in
test/ttlang/Dialect/TTKernel/Transforms/specialize_cores.mlir.
"""

import torch
import ttnn

import ttl
from ttlang_test_utils import assert_pcc, require_hardware, to_dram

TILE = 32
GRID_X, GRID_Y = 2, 2


@ttl.operation(grid=(GRID_X, GRID_Y))
def branch_swap(a, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute_fn():
        with a_dfb.wait() as a_tile, out_dfb.reserve() as o:
            o.store(a_tile)

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.node(dims=2)
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[y, 0], blk)
            if x == 0:
                tx = ttl.copy(a[y, 1], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[y, x])
            tx.wait()


# No emitc.if / scf.if survives anywhere (asserted via --implicit-check-not on
# the RUN line): every clone's coordinate branch is const-folded.
#
# core_x == 0 takes the branch, so its clone reads column 0 then column 1 --
# two async_reads.
# CHECK-LABEL: func.func @dm_read_c0_0
# CHECK-SAME:    ttl.core_coord = {{\[\[}}0, 0]]
# CHECK:         async_read(
# CHECK:         async_read(
# CHECK:         return
#
# core_x == 1 does not take the branch, so the column-1 read is eliminated --
# a single async_read remains (CHECK-NOT bounded by the trailing return).
# CHECK-LABEL: func.func @dm_read_c1_0
# CHECK-SAME:    ttl.core_coord = {{\[\[}}1, 0]]
# CHECK:         async_read(
# CHECK-NOT:     async_read(
# CHECK:         return

# RUNTIME: specialize-cores correctness OK

if __name__ == "__main__":
    require_hardware()
    device = ttnn.open_device(device_id=0)
    try:
        shape = (GRID_Y * TILE, GRID_X * TILE)
        a_torch = torch.randn(shape, dtype=torch.bfloat16)
        # Reference: swap the two tile-columns (GRID_X == 2).
        expected = torch.cat([a_torch[:, TILE:], a_torch[:, :TILE]], dim=1).contiguous()

        a = to_dram(a_torch, device)
        out = to_dram(torch.zeros(shape, dtype=torch.bfloat16), device)
        branch_swap(a, out, options="--ttl-specialize-cores")

        assert_pcc(expected, ttnn.to_torch(out))
        print("specialize-cores correctness OK")
    finally:
        ttnn.close_device(device)
