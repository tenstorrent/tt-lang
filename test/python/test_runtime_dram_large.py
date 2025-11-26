# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s > %t.output.txt 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-OUTPUT < %t.output.txt

# Runtime test for large DRAM tensors using tilize-on-the-fly.
# 1024x1024 f32 = 4MB (exceeds L1 ~1.5MB per core).

import torch
from ttlang.d2m_api import *


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)], num_outs=1)
def test_runtime_dram_large(lhs, rhs, lhs_tiled, rhs_tiled, out):
    lhs_accessor = TensorAccessor(lhs, memory_space="DRAM")
    rhs_accessor = TensorAccessor(rhs, memory_space="DRAM")

    @compute()
    def add_compute(
        lhs_scalar_cb: CircularBuffer,
        rhs_scalar_cb: CircularBuffer,
        lhs_tiled_cb: CircularBuffer,
        rhs_tiled_cb: CircularBuffer,
        out_cb: CircularBuffer
    ):
        l_scalar = lhs_scalar_cb.wait()
        r_scalar = rhs_scalar_cb.wait()

        l_tiled = lhs_tiled_cb.reserve()
        r_tiled = rhs_tiled_cb.reserve()

        tilize(l_scalar, l_tiled)
        tilize(r_scalar, r_tiled)

        lhs_tiled_cb.push()
        rhs_tiled_cb.push()
        lhs_scalar_cb.pop()
        rhs_scalar_cb.pop()

        l_tile = lhs_tiled_cb.wait()
        r_tile = rhs_tiled_cb.wait()

        o = out_cb.reserve()
        result = l_tile + r_tile
        o.store(result)
        out_cb.push()

        lhs_tiled_cb.pop()
        rhs_tiled_cb.pop()

    @datamovement()
    def dm_lhs(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
               lhs_tiled_cb: CircularBuffer, rhs_tiled_cb: CircularBuffer,
               out_cb: CircularBuffer):
        lhs_shard = lhs_scalar_cb.reserve()
        tx = dma(lhs_accessor[0, 0], lhs_shard)
        tx.wait()
        lhs_scalar_cb.push()

    @datamovement()
    def dm_rhs(lhs_scalar_cb: CircularBuffer, rhs_scalar_cb: CircularBuffer,
               lhs_tiled_cb: CircularBuffer, rhs_tiled_cb: CircularBuffer,
               out_cb: CircularBuffer):
        rhs_shard = rhs_scalar_cb.reserve()
        tx = dma(rhs_accessor[0, 0], rhs_shard)
        tx.wait()
        rhs_scalar_cb.push()

    return Program(add_compute, dm_lhs, dm_rhs)(lhs, rhs, lhs_tiled, rhs_tiled, out)


lhs = torch.full((1024, 1024), 2.0)
rhs = torch.full((1024, 1024), 3.0)
lhs_tiled = torch.zeros((32, 32))
rhs_tiled = torch.zeros((32, 32))
out = torch.full((32, 32), -999.0)

print("=== BEFORE KERNEL ===")
print(f"Tensor size: 1024x1024 f32 = 4MB (exceeds L1 ~1.5MB)")
print(f"lhs: all 2.0, rhs: all 3.0, out: all -999.0")
print(f"Expected: all 5.0")

test_runtime_dram_large(lhs, rhs, lhs_tiled, rhs_tiled, out)

print("\n=== AFTER KERNEL ===")
# CHECK-OUTPUT: === AFTER KERNEL ===
print(f"out[0, 0] = {out[0, 0].item()}")
print(
    f"out min/max/mean: {out.min().item():.1f} / {out.max().item():.1f} / {out.mean().item():.1f}"
)

expected = torch.full((32, 32), 5.0)
if torch.allclose(out, expected, rtol=1e-2, atol=1e-2):
    print("PASS: Output matches expected (2.0 + 3.0 = 5.0)")
    # CHECK-OUTPUT: PASS: Output matches expected
else:
    print(
        f"FAIL: Expected all 5.0, got values from {out.min().item()} to {out.max().item()}"
    )
