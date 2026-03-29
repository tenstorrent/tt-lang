# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Simple transpose kernel -- verifies transpose lowers to correct TTL ops
and generates correct C++ (transpose_wh_init, transpose_wh_tile).
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def transpose_kernel(inp, out):
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute_fn():
        with inp_dfb.wait() as inp_blk, out_dfb.reserve() as out_blk:
            result = ttl.math.transpose(inp_blk)
            out_blk.store(result)

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx_inp = ttl.copy(inp[0, 0], inp_blk)
        tx_inp.wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx_out = ttl.copy(out_blk, out[0, 0])
        tx_out.wait()
        out_blk.pop()


# Initial MLIR: verify transpose op is present.
# CHECK: ttl.transpose

# Generated C++: verify transpose_wh_init and transpose_wh_tile.
# CHECK-CPP: transpose_wh_init
# CHECK-CPP: transpose_wh_tile

if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_l1

    device = ttnn.open_device(device_id=0)

    try:
        inp = to_l1(torch.ones(32, 32, dtype=torch.bfloat16), device)
        out = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)

        transpose_kernel(inp, out)
    finally:
        ttnn.close_device(device)
