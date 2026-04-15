# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output
# RUN: FileCheck %s --check-prefix=CHECK-RESULT < %t.output

"""
Multi-tile row-wise softmax with compiler-allocated intermediate DFBs.

Modeled after _examples/softmax.py but without manually declared intermediate
DFBs. The user provides only inp, scaler, and out DFBs. The compiler
inserts scratch DFBs for the reduce_max result, exp(x - max), and
reduce_sum result via ttl-insert-intermediate-dfbs.

Verifies generated C++ (multiple sync regions with scratch DFB push/wait)
and runtime correctness (PCC > 0.99 against torch.softmax).
"""

import torch
import ttnn
import ttl
from ttlang_test_utils import to_l1

TILE = 32
ROWS = 2
COLS = 4


@ttl.operation(grid=(1, 1))
def softmax_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(ROWS, COLS), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(ROWS, COLS), block_count=2)

    @ttl.compute()
    def compute():
        with inp_dfb.wait() as x_blk, scaler_dfb.wait() as s_blk:
            mx = ttl.math.reduce_max(x_blk, s_blk, dims=[1])
            shifted = ttl.sub(x_blk, ttl.math.broadcast(mx, x_blk, dims=[1]))
            ex = ttl.exp(shifted)
            sm = ttl.math.reduce_sum(ex, s_blk, dims=[1])
            inv_sum = ttl.recip(ttl.math.broadcast(sm, ex, dims=[1]))
            with out_dfb.reserve() as out_blk:
                out_blk.store(ttl.mul(ex, inv_sum))

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            ttl.copy(inp[0:ROWS, 0:COLS], blk).wait()
        with scaler_dfb.reserve() as blk:
            ttl.copy(scaler[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0:ROWS, 0:COLS]).wait()


# =============================================================================
# C++ Checks - Verify scratch DFB push/wait pattern in compute kernel.
# The compiler creates scratch DFBs for reduce_max result, exp values,
# and reduce_sum result. Each gets a cb_push_back followed by cb_wait_front.
# =============================================================================

# CHECK-CPP: // compute
# CHECK-CPP: void kernel_main()

# reduce_max -> pack to scratch DFB, push, then wait for bcast.
# CHECK-CPP: reduce_tile<PoolType::MAX
# CHECK-CPP: pack_tile
# CHECK-CPP: cb_push_back
# CHECK-CPP: cb_wait_front

# bcast(max) + sub + exp -> pack to scratch DFB, push, then wait for reduce_sum.
# CHECK-CPP: unary_bcast
# CHECK-CPP: exp_tile
# CHECK-CPP: pack_tile
# CHECK-CPP: cb_push_back
# CHECK-CPP: cb_wait_front

# reduce_sum -> pack to scratch DFB, push, then wait for final bcast.
# CHECK-CPP: reduce_tile<PoolType::SUM
# CHECK-CPP: pack_tile
# CHECK-CPP: cb_push_back
# CHECK-CPP: cb_wait_front

# bcast(sum) + recip + mul -> pack to output DFB.
# CHECK-CPP: unary_bcast
# CHECK-CPP: recip_tile
# CHECK-CPP: mul_binary_tile
# CHECK-CPP: pack_tile

# =============================================================================
# Runtime result check
# =============================================================================

# CHECK-RESULT: RESULT OK

device = ttnn.open_device(device_id=0)

inp_torch = torch.randn(ROWS * TILE, COLS * TILE, dtype=torch.bfloat16)
scaler_torch = torch.ones(TILE, TILE, dtype=torch.bfloat16)
out_torch = torch.zeros(ROWS * TILE, COLS * TILE, dtype=torch.bfloat16)

inp = to_l1(inp_torch, device)
scaler = to_l1(scaler_torch, device)
out = to_l1(out_torch, device)

softmax_kernel(inp, scaler, out)
result = ttnn.to_torch(out).float()

expected = torch.softmax(inp_torch.float(), dim=-1)

pcc = torch.corrcoef(torch.stack([result.flatten(), expected.flatten()]))[0, 1].item()

print(f"PCC: {pcc:.6f}")
if pcc > 0.99:
    print("RESULT OK")
else:
    print(f"RESULT FAIL: PCC {pcc} below 0.99")

ttnn.close_device(device)
