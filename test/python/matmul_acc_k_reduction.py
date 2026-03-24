# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Matmul with K-dimension accumulation: accumulates partial matrix products
over the inner dimension using L1 accumulation. Each iteration computes a
partial matmul and accumulates via acc=True:

    out = sum_k(A[:, k] @ B[k, :])
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch

TILE_SIZE = 32
K_TILES = 3


@ttl.kernel(grid=(1, 1))
def matmul_k_acc_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with out_dfb.reserve() as o:
            # First K slice: overwrite.
            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                o.store(a_blk @ b_blk)
            # Remaining K slices: accumulate via L1.
            for k in range(K_TILES - 1):
                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                    o.store(a_blk @ b_blk, acc=True)

    @ttl.datamovement()
    def dm_read():
        for k in range(K_TILES):
            with a_dfb.reserve() as blk:
                tx = ttl.copy(a[0:1, k : k + 1], blk)
                tx.wait()
            with b_dfb.reserve() as blk:
                tx = ttl.copy(b[k : k + 1, 0:1], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()
            blk.pop()


# Initial IR: first store is non-acc matmul, loop stores are acc=true matmul.
# CHECK-LABEL: func.func @compute_fn
# CHECK:         ttl.matmul
# CHECK:         ttl.store
# CHECK-NOT:     acc = true
# CHECK:         ttl.matmul
# CHECK:         ttl.store
# CHECK-SAME:    acc = true

# C++ output: L1 accumulation on loop body stores.
# CHECK-CPP: // compute
# CHECK-CPP: experimental::matmul_block(
# CHECK-CPP: pack_tile
# CHECK-CPP: experimental::matmul_block(
# CHECK-CPP: llk_pack_reconfig_l1_acc(
# CHECK-CPP-NEXT: pack_tile
# CHECK-CPP-NEXT: llk_pack_reconfig_l1_acc(

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        M, K, N = TILE_SIZE, K_TILES * TILE_SIZE, TILE_SIZE

        a_torch = torch.randn((M, K), dtype=torch.bfloat16)
        b_torch = torch.randn((K, N), dtype=torch.bfloat16)
        out_torch = torch.zeros((M, N), dtype=torch.bfloat16)

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
        out = to_device(out_torch)

        matmul_k_acc_kernel(a, b, out)

        result = ttnn.to_torch(out)
        golden = a_torch @ b_torch
        pcc = torch.corrcoef(
            torch.stack([result.flatten().float(), golden.flatten().float()])
        )[0, 1].item()
        if pcc > 0.999:
            print("PASS")
        else:
            print(f"FAIL: PCC {pcc:.6f} < 0.999")

    finally:
        ttnn.close_device(device)
