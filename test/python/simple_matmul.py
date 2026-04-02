# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: env TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s --no-ttl-maximize-dst --no-ttl-fpu-binary-ops > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Standalone matmul: single 1x1 tile multiply through the full pipeline.
Verifies ttl.matmul in initial IR and matmul_block in C++ output.
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch


@ttl.operation(grid=(1, 1))
def matmul_kernel(a, b, out):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def mm_compute():
        a_blk = a_dfb.wait()
        b_blk = b_dfb.wait()
        o_blk = out_dfb.reserve()
        result = a_blk @ b_blk
        o_blk.store(result)
        a_blk.pop()
        b_blk.pop()
        o_blk.push()

    @ttl.datamovement()
    def dm_read():
        a_blk = a_dfb.reserve()
        tx_a = ttl.copy(a[0, 0], a_blk)
        tx_a.wait()
        a_blk.push()

        b_blk = b_dfb.reserve()
        tx_b = ttl.copy(b[0, 0], b_blk)
        tx_b.wait()
        b_blk.push()

    @ttl.datamovement()
    def dm_write():
        o_blk = out_dfb.wait()
        tx_o = ttl.copy(o_blk, out[0, 0])
        tx_o.wait()
        o_blk.pop()


# =============================================================================
# Initial IR: ttl.matmul emitted by __matmul__
# =============================================================================

# CHECK-LABEL: func.func @mm_compute
# CHECK: ttl.matmul
# CHECK: ttl.store


# =============================================================================
# C++ output: matmul_block init and compute
# =============================================================================

# CHECK-CPP: mm_block_init(
# CHECK-CPP: matmul_block(
# CHECK-CPP: pack_tile

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        a_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        b_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.from_torch(
            b_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        a = ttnn.to_memory_config(a, memory_config=ttnn.L1_MEMORY_CONFIG)
        b = ttnn.to_memory_config(b, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        matmul_kernel(a, b, out)

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
