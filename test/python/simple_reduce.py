# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple reduce kernel -- verifies reduce_sum lowers to correct TTL ops
and generates correct C++ (reduce_init, reduce_tile, reduce_uninit).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def reduce_kernel(inp, scaler, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with (
            inp_dfb.wait() as inp_blk,
            scaler_dfb.wait() as scaler_blk,
            out_dfb.reserve() as out_blk,
        ):
            result = ttl.math.reduce_sum(inp_blk, scaler_blk, dims=[0, 1])
            out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[0, 0], inp_blk)
        tx_inp.wait()
        inp_blk.push()
        scaler_blk = scaler_dfb.reserve()
        tx_scaler = ttl.copy(scaler[0, 0], scaler_blk)
        tx_scaler.wait()
        scaler_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[0, 0])
        tx_out.wait()
        out_blk.pop()


# Initial MLIR: verify reduce op with correct dims and reduce_type.
# CHECK: ttl.reduce
# CHECK-SAME: 0 : i32
# CHECK-SAME: [0, 1]

# Generated C++: verify reduce_init, reduce_tile, reduce_uninit sequence.
# CHECK-CPP: reduce_init
# CHECK-CPP: reduce_tile
# CHECK-CPP: reduce_uninit

if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_l1

    device = ttnn.open_device(device_id=0)

    try:
        scaler_t = torch.zeros(32, 32, dtype=torch.bfloat16)
        scaler_t[0, :] = 1.0
        scaler_t[16, :] = 1.0

        inp = to_l1(torch.ones(32, 32, dtype=torch.bfloat16), device)
        scaler = to_l1(scaler_t, device)
        out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

        reduce_kernel(inp, scaler, out)
    finally:
        ttnn.close_device(device)
