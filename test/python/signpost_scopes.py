# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.output

"""
Matmul + accumulate kernel - verifies user-defined signpost scopes in generated
C++ compute kernel. Computes C += A @ B with separate signpost regions for
matmul vs accumulate to enable per-region profiling.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)


@ttl.kernel(grid=(1, 1))
def matmul_accum_kernel(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 2), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 1), buffer_factor=2)
    ab_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with (
            a_dfb.wait() as a_blk,
            b_dfb.wait() as b_blk,
            ab_dfb.reserve() as ab_blk,
        ):
            with ttl.signpost("matmul"):
                ab_blk.store(a_blk @ b_blk)

        with (
            ab_dfb.wait() as ab_blk,
            c_dfb.wait() as c_blk,
            out_dfb.reserve() as out_blk,
        ):
            with ttl.signpost("accumulate"):
                out_blk.store(ab_blk + c_blk)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0:1, 0:2], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0:2, 0:1], blk)
            tx.wait()
        with c_dfb.reserve() as blk:
            tx = ttl.copy(c[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, c[0, 0])
            tx.wait()


# =============================================================================
# C++ Kernel Checks - Verify generated compute kernel structure
# =============================================================================

# CHECK: // compute_fn
# CHECK: void kernel_main()
# CHECK: DeviceZoneScopedN("ttl_matmul");
# CHECK: mm_init(
# CHECK: matmul_tiles(
# CHECK: pack_tile
# CHECK: DeviceZoneScopedN("ttl_accumulate");
# CHECK: add_binary_tile_init();
# CHECK: add_binary_tile(
# CHECK: pack_tile


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        a = torch.randn(32, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 32, dtype=torch.bfloat16)
        c = torch.zeros(32, 32, dtype=torch.bfloat16)

        def from_torch(tensor):
            return ttnn.from_torch(
                tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        a_tt = from_torch(a)
        b_tt = from_torch(b)
        c_tt = from_torch(c)

        matmul_accum_kernel(a_tt, b_tt, c_tt)

    finally:
        ttnn.close_device(device)
