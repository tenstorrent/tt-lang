# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: elementwise op feeding directly into reduce is not yet
supported (#474). The compiler should emit a clear error instead of the
cryptic "failed to legalize operation 'ttl.mul'" message.

Workaround: store the elementwise result to a dataflow buffer first.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl

TILE = 32
HEAD_TILES = 4


# CHECK: elementwise operations feeding into reduce cannot be fused yet; store the intermediate result to a dataflow buffer before passing it to reduce
# CHECK:   --> {{.*}}invalid_reduce_fused_elementwise.py:[[LINE:[0-9]+]]:{{[0-9]+}}
# CHECK:    |
# CHECK: [[LINE]] |                     o.store(ttl.math.reduce_sum(av * bv, sc, dims=[0, 1]))
# CHECK:    |                         ^
@ttl.operation(grid=(1, 1))
def fused_reduce_kernel(inp_a, inp_b, scaler, out):
    a_dfb = ttl.make_dataflow_buffer_like(inp_a, shape=(1, HEAD_TILES), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(inp_b, shape=(1, HEAD_TILES), block_count=2)
    sc_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with sc_dfb.wait() as sc:
            with a_dfb.wait() as av, b_dfb.wait() as bv:
                with out_dfb.reserve() as o:
                    o.store(ttl.math.reduce_sum(av * bv, sc, dims=[0, 1]))

    @ttl.datamovement()
    def dm_read():
        with sc_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()
        with a_dfb.reserve() as blk:
            tx = ttl.copy(inp_a[0:1, 0:HEAD_TILES], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(inp_b[0:1, 0:HEAD_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, 0])
            tx.wait()


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import to_dram, to_l1

    device = ttnn.open_device(device_id=0)

    try:
        hd = HEAD_TILES * TILE
        inp_a = to_dram(torch.randn(TILE, hd, dtype=torch.bfloat16), device)
        inp_b = to_dram(torch.randn(TILE, hd, dtype=torch.bfloat16), device)
        scaler = to_l1(torch.ones(TILE, TILE, dtype=torch.bfloat16), device)
        out = to_dram(torch.zeros(TILE, TILE, dtype=torch.bfloat16), device)

        fused_reduce_kernel(inp_a, inp_b, scaler, out)

        print("ERROR: Expected compilation error was not raised!")
        exit(1)
    finally:
        ttnn.close_device(device)
