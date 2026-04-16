# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Fused matmul + bias + SiLU: store(silu(A @ B + bias)).

Models the SwiGLU gate path in transformer FFNs with a shape that triggers
DST subblocking: A[4x6] @ B[6x4] = C[4x4] (16 output tiles, bf16 DST
capacity = 8). The subblocking pass splits the 4x4 output into 1x4
subblocks, each with its own sync region. Verifies initial IR structure,
generated C++ op sequence (including the subblock loop), and numerical
correctness on hardware.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch

TILE = 32
M_BLK, K_BLK, N_BLK = 4, 6, 4


@ttl.operation(grid=(1, 1))
def swiglu_gate_kernel(a_tensor, b_tensor, bias_tensor, out_tensor):
    a_dfb = ttl.make_dataflow_buffer_like(a_tensor, shape=(M_BLK, K_BLK), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_tensor, shape=(K_BLK, N_BLK), block_count=2)
    bias_dfb = ttl.make_dataflow_buffer_like(
        bias_tensor, shape=(M_BLK, N_BLK), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out_tensor, shape=(M_BLK, N_BLK), block_count=2
    )

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            bias_dfb.wait() as bi_blk,
        ):
            with out_dfb.reserve() as o_blk:
                o_blk.store(ttl.silu(a_blk @ b_blk + bi_blk))

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            ttl.copy(a_tensor[0:M_BLK, 0:K_BLK], blk).wait()
        with b_dfb.reserve() as blk:
            ttl.copy(b_tensor[0:K_BLK, 0:N_BLK], blk).wait()
        with bias_dfb.reserve() as blk:
            ttl.copy(bias_tensor[0:M_BLK, 0:N_BLK], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out_tensor[0:M_BLK, 0:N_BLK]).wait()


# =============================================================================
# Initial IR: matmul feeds into add (bias), then silu, then store.
# Shapes: A[4x6] @ B[6x4] -> [4x4], bias[4x4].
# =============================================================================

# CHECK-LABEL: func.func @compute_fn
# CHECK:         %[[A:.*]] = ttl.attach_cb
# CHECK:         %[[B:.*]] = ttl.attach_cb
# CHECK:         %[[BI:.*]] = ttl.attach_cb
# CHECK:         %[[MM:.*]] = ttl.matmul %[[A]], %[[B]]
# CHECK-SAME:      tensor<4x6x!ttcore.tile<32x32, bf16>>
# CHECK:         %[[ADD:.*]] = ttl.add %[[MM]], %[[BI]]
# CHECK:         %[[SILU:.*]] = ttl.silu %[[ADD]]
# CHECK:         ttl.store %[[SILU]]


# =============================================================================
# C++ output: 4x4 output (16 tiles) exceeds bf16 DST capacity (8), so
# subblocking splits into an outer loop over M rows. Each iteration
# processes a 1x4 subblock: bias preload, K-loop matmul, SiLU, pack.
# =============================================================================

# CHECK-CPP:       mm_block_init(
#   Outer subblock loop over M rows.
# CHECK-CPP:       for
# CHECK-CPP:         tile_regs_acquire
#   Bias preload: 4 copy_tile ops per subblock row.
# CHECK-CPP:         copy_tile_init(
# CHECK-CPP:         copy_tile(
# CHECK-CPP:         copy_tile(
# CHECK-CPP:         copy_tile(
# CHECK-CPP:         copy_tile(
#   Matmul with K-loop (kt=6).
# CHECK-CPP:         mm_block_init_short(
# CHECK-CPP:         for
# CHECK-CPP:           matmul_block(
#   SiLU on each of the 4 subblock tiles.
# CHECK-CPP:         silu_tile_init(
# CHECK-CPP-NEXT:    silu_tile(
# CHECK-CPP-NEXT:    silu_tile(
# CHECK-CPP-NEXT:    silu_tile(
# CHECK-CPP-NEXT:    silu_tile(
#   Pack 4 tiles from this subblock.
# CHECK-CPP:         tile_regs_commit
# CHECK-CPP-NEXT:    tile_regs_wait
# CHECK-CPP:         pack_tile
# CHECK-CPP:         pack_tile
# CHECK-CPP:         pack_tile
# CHECK-CPP:         pack_tile
# CHECK-CPP:         tile_regs_release
#   No explicit add -- folded into matmul accumulator.
# CHECK-CPP-NOT:   add_tiles
# CHECK-CPP-NOT:   add_binary_tile

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        torch.manual_seed(42)
        total_m = M_BLK * TILE
        total_k = K_BLK * TILE
        total_n = N_BLK * TILE

        a_torch = torch.randn((total_m, total_k), dtype=torch.bfloat16)
        b_torch = torch.randn((total_k, total_n), dtype=torch.bfloat16)
        bias_torch = torch.randn((total_m, total_n), dtype=torch.bfloat16)
        out_torch = torch.zeros((total_m, total_n), dtype=torch.bfloat16)

        to_device = lambda t: ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        a_dev = to_device(a_torch)
        b_dev = to_device(b_torch)
        bias_dev = to_device(bias_torch)
        out_dev = to_device(out_torch)

        swiglu_gate_kernel(a_dev, b_dev, bias_dev, out_dev)

        result = ttnn.to_torch(out_dev).float()
        golden = torch.nn.functional.silu(
            a_torch.float() @ b_torch.float() + bias_torch.float()
        )

        pcc = torch.corrcoef(torch.stack([result.flatten(), golden.flatten()]))[
            0, 1
        ].item()
        if pcc > 0.99:
            print("PASS")
        else:
            print(f"FAIL: PCC {pcc:.6f} < 0.99")

    finally:
        ttnn.close_device(device)
