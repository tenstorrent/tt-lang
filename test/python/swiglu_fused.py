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

Models the SwiGLU gate path in transformer FFNs with a non-trivial shape:
A[2x4] @ B[4x2] = C[2x2] (4 output tiles, K=4). The matmul has a K-loop,
the bias folds into the accumulator, and SiLU is applied per-tile after
the M*N expansion. Verifies initial IR structure, generated C++ op
sequence, and numerical correctness on hardware.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch

TILE = 32
M_BLK, K_BLK, N_BLK = 2, 4, 2


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
# Shapes: A[2x4] @ B[4x2] -> [2x2], bias[2x2].
# =============================================================================

# CHECK-LABEL: func.func @compute_fn
# CHECK:         %[[A:.*]] = ttl.attach_cb
# CHECK:         %[[B:.*]] = ttl.attach_cb
# CHECK:         %[[BI:.*]] = ttl.attach_cb
# CHECK:         %[[MM:.*]] = ttl.matmul %[[A]], %[[B]]
# CHECK-SAME:      tensor<2x4x!ttcore.tile<32x32, bf16>>
# CHECK:         %[[ADD:.*]] = ttl.add %[[MM]], %[[BI]]
# CHECK:         %[[SILU:.*]] = ttl.silu %[[ADD]]
# CHECK:         ttl.store %[[SILU]]


# =============================================================================
# C++ output: bias preloaded via copy_tile (4 tiles for 2x2 output),
# matmul_block with K-loop (kt=4), SiLU on each DST tile, then 4 pack_tiles.
# =============================================================================

# CHECK-CPP:       mm_block_init(
# CHECK-CPP:       tile_regs_acquire
#   Bias preload: 4 copy_tile ops for the 2x2 output.
# CHECK-CPP:       copy_tile_init(
# CHECK-CPP:       copy_tile(
# CHECK-CPP:       copy_tile(
# CHECK-CPP:       copy_tile(
# CHECK-CPP:       copy_tile(
#   Matmul with K-loop (kt=4).
# CHECK-CPP:       mm_block_init_short(
# CHECK-CPP:       for
# CHECK-CPP:         matmul_block(
#   SiLU on each of the 4 output tiles.
# CHECK-CPP:       silu_tile_init(
# CHECK-CPP-NEXT:  silu_tile(
# CHECK-CPP-NEXT:  silu_tile(
# CHECK-CPP-NEXT:  silu_tile(
# CHECK-CPP-NEXT:  silu_tile(
#   Pack (combined into pack_tile_block for contiguous DST).
# CHECK-CPP:       tile_regs_commit
# CHECK-CPP-NEXT:  tile_regs_wait
# CHECK-CPP-NEXT:  pack_tile_block(
# CHECK-CPP-NEXT:  tile_regs_release
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
