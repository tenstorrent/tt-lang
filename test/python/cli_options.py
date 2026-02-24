# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s --no-maximize-dst > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Tests that compiler options are correctly propagated from sys.argv.

Runs a 2-tile unary kernel with --no-maximize-dst passed via CLI.
Without maximize-dst, each tile gets its own sync region (acquire/commit
pair), so tile_regs_commit appears between the two sigmoid_tile calls.

The default maximize-dst path (single sync region for both tiles) is
already covered by simple_add_multitile.py.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(1, 1))
def sigmoid_2tile_kernel(inp, out):
    """Sigmoid on a 1x2 tile block (2 tiles total)."""
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 2), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 2), buffer_factor=2)

    @ttl.compute()
    def compute_sigmoid():
        i = inp_dfb.wait()
        o = out_dfb.reserve()
        result = ttl.math.sigmoid(i)
        o.store(result)
        i.pop()
        o.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_dfb.reserve()
        tx = ttl.copy(inp[0, 0:2], inp_blk)
        tx.wait()
        inp_blk.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_dfb.wait()
        tx = ttl.copy(out_blk, out[0, 0:2])
        tx.wait()
        out_blk.pop()


# =============================================================================
# No maximize-dst (via CLI --no-maximize-dst): one sync region per tile.
# Only ONE sigmoid_tile appears between acquire and commit (the loop runs
# once per tile). With maximize-dst there would be TWO sigmoid_tile calls
# between acquire and commit. CHECK-NOT verifies the CLI option took effect.
# =============================================================================

# CHECK-CPP: // compute_sigmoid
# CHECK-CPP: tile_regs_acquire
# CHECK-CPP: sigmoid_tile(
# CHECK-CPP-NOT: sigmoid_tile(
# CHECK-CPP: tile_regs_commit


if __name__ == "__main__":
    import torch

    device = ttnn.open_device(device_id=0)

    try:
        # 32x64 = 1x2 tiles of 32x32
        inp_torch = torch.rand((32, 64), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 64), dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
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

        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        print("Compiling sigmoid 2-tile kernel...")
        sigmoid_2tile_kernel(inp, out)

        print("=== CLI Options Test Complete ===")

    finally:
        ttnn.close_device(device)
