# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Multinode row-wise softmax with compiler-allocated intermediate DFBs.

Each core in the (COLS, ROWS) grid processes one row of COLS tiles.
The user provides only inp, scaler, and out DFBs. The compiler inserts
intermediate DFBs for reduce_max, exp(x - max), and reduce_sum results
via ttl-insert-intermediate-dfbs.

Verifies generated C++ (DFB push/wait pattern) and runtime
correctness (PCC > 0.95 against torch.softmax).
"""

import ttl

try:
    import ttnn
except ImportError:
    print("TTNN not available - exiting")
    exit(0)

import torch
from ttlang_test_utils import assert_pcc, to_dram, to_l1

TILE = 32
ROWS = 2
COLS = 4


@ttl.operation(grid=(COLS, ROWS))
def softmax_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x_blk, scaler_dfb.wait() as s_blk:
            mx = ttl.math.reduce_max(x_blk, s_blk, dims=[0, 1])
            shifted = ttl.sub(x_blk, ttl.math.broadcast(mx, x_blk, dims=[0, 1]))
            ex = ttl.exp(shifted)
            sm = ttl.math.reduce_sum(ex, s_blk, dims=[0, 1])
            inv_sum = ttl.recip(ttl.math.broadcast(sm, ex, dims=[0, 1]))
            with out_dfb.reserve() as out_blk:
                out_blk.store(ttl.mul(ex, inv_sum))

    @ttl.datamovement()
    def dm_read():
        col, row = ttl.node(dims=2)
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[row, col], blk).wait()
        with scaler_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        col, row = ttl.node(dims=2)
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[row, col]).wait()


# =============================================================================
# C++ Checks - Verify intermediate DFB push/wait pattern in compute kernel.
# =============================================================================

# CHECK-CPP: // compute
# CHECK-CPP: void kernel_main()

# reduce_max -> pack to intermediate DFB, push, wait for bcast.
# CHECK-CPP: reduce_tile<PoolType::MAX
# CHECK-CPP: pack_tile
# CHECK-CPP: cb_push_back
# CHECK-CPP: cb_wait_front

# bcast(max) + sub + exp -> pack to intermediate DFB, push, wait for reduce_sum.
# CHECK-CPP: unary_bcast
# CHECK-CPP: exp_tile
# CHECK-CPP: pack_tile
# CHECK-CPP: cb_push_back
# CHECK-CPP: cb_wait_front

# reduce_sum -> pack to intermediate DFB, push, wait for final bcast.
# CHECK-CPP: reduce_tile<PoolType::SUM
# CHECK-CPP: pack_tile
# CHECK-CPP: cb_push_back
# CHECK-CPP: cb_wait_front

# bcast(sum) + recip + mul -> pack to output.
# CHECK-CPP: unary_bcast
# CHECK-CPP: recip_tile
# CHECK-CPP: mul_binary_tile
# CHECK-CPP: pack_tile

# =============================================================================
# Runtime result check
# =============================================================================

# CHECK-RESULT: PASS

if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.randn(ROWS * TILE, COLS * TILE, dtype=torch.bfloat16)
        scaler_torch = torch.ones(TILE, TILE, dtype=torch.bfloat16)
        out_torch = torch.zeros(ROWS * TILE, COLS * TILE, dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        scaler = to_l1(scaler_torch, device)
        out = to_dram(out_torch, device)

        softmax_kernel(inp, scaler, out)
        result = ttnn.to_torch(out).float()

        # Scalar softmax per tile: reduce_max/reduce_sum use dims=[0,1],
        # so each 32x32 tile is normalized over all 1024 elements.
        expected = torch.zeros_like(inp_torch, dtype=torch.float32)
        for row_idx in range(ROWS):
            for col_idx in range(COLS):
                r0, r1 = row_idx * TILE, (row_idx + 1) * TILE
                c0, c1 = col_idx * TILE, (col_idx + 1) * TILE
                tile = inp_torch[r0:r1, c0:c1].float()
                expected[r0:r1, c0:c1] = torch.softmax(tile.flatten(), dim=0).reshape(
                    TILE, TILE
                )

        # Six chained bf16 operations (reduce_max, sub, exp, reduce_sum,
        # recip, mul) each truncate to bf16, compounding precision loss.
        # Measured PCC ~0.96 on Blackhole.
        assert_pcc(expected, result, threshold=0.999)
        print("PASS")

    finally:
        ttnn.close_device(device)
