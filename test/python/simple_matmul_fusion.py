# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Matmul with accumulation: store((a @ b) + c) lowers to copy_tile(c) +
matmul_block(DST += A*B), eliminating the explicit add. Verifies initial IR,
generated C++ op sequence, and numerical correctness on hardware.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch


@ttl.operation(grid=(1, 1))
def matmul_fusion_kernel(a, b, c, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        c_blk = c_dfb.wait()
        o_blk = out_dfb.reserve()
        o_blk.store((a_blk @ b_blk) + c_blk)
        a_blk.pop()
        b_blk.pop()
        c_blk.pop()
        o_blk.push()

    @ttl.datamovement()
    def dm_read():
        a_blk = a_dfb.reserve()
        tx = ttl.copy(a[0, 0], a_blk)
        tx.wait()
        a_blk.push()

        b_blk = b_dfb.reserve()
        tx = ttl.copy(b[0, 0], b_blk)
        tx.wait()
        b_blk.push()

        c_blk = c_dfb.reserve()
        tx = ttl.copy(c[0, 0], c_blk)
        tx.wait()
        c_blk.push()

    @ttl.datamovement()
    def dm_write():
        o_blk = out_dfb.wait()
        tx = ttl.copy(o_blk, out[0, 0])
        tx.wait()
        o_blk.pop()


# =============================================================================
# Initial IR: ttl.matmul feeds into ttl.add, result stored via ttl.store.
# After the pipeline, the add vanishes — matmul_block accumulates onto the
# copy_tile'd bias.
# =============================================================================

# CHECK-LABEL: func.func @compute_fn
# CHECK:         %[[A:.*]] = ttl.attach_cb
# CHECK:         %[[B:.*]] = ttl.attach_cb
# CHECK:         %[[C:.*]] = ttl.attach_cb
# CHECK:         %[[MM:.*]] = ttl.matmul %[[A]], %[[B]]
# CHECK-SAME:      tensor<1x1x!ttcore.tile<32x32, bf16>>
# CHECK:         %[[SUM:.*]] = ttl.add %[[MM]], %[[C]]
# CHECK:         ttl.store %[[SUM]]


# =============================================================================
# C++ output: add vanishes — copy_tile pre-loads bias into DST, matmul
# accumulates (DST += A*B), then pack. No add_tiles or add_binary_tile.
# =============================================================================

# CHECK-CPP:       mm_block_init(
# CHECK-CPP:       tile_regs_acquire
# CHECK-CPP-NEXT:  copy_tile_init(
# CHECK-CPP-NEXT:  copy_tile(
# CHECK-CPP-NEXT:  mm_block_init_short(
# CHECK-CPP-NEXT:  experimental::matmul_block(
# CHECK-CPP-NEXT:  tile_regs_commit
# CHECK-CPP-NEXT:  tile_regs_wait
# CHECK-CPP-NEXT:  pack_tile
# CHECK-CPP-NEXT:  tile_regs_release
# CHECK-CPP-NOT:   add_tiles
# CHECK-CPP-NOT:   add_binary_tile

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        a_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        b_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        c_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        to_device = lambda t: ttnn.to_memory_config(
            ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        a = to_device(a_torch)
        b = to_device(b_torch)
        c = to_device(c_torch)
        out = to_device(out_torch)

        matmul_fusion_kernel(a, b, c, out)

        result = ttnn.to_torch(out)
        golden = (a_torch @ b_torch) + c_torch
        pcc = torch.corrcoef(
            torch.stack([result.flatten().float(), golden.flatten().float()])
        )[0, 1].item()
        if pcc > 0.999:
            print("PASS")
        else:
            print(f"FAIL: PCC {pcc:.6f} < 0.999")

    finally:
        ttnn.close_device(device)
