# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Scatter-Compute pattern - one source broadcasts to multiple workers.

Core (0,0) reads input and scatters (broadcasts) to all 4 cores.
Each core applies a different function to the same input data:
  Core 0: abs(x)
  Core 1: neg(x)
  Core 2: exp(x)
  Core 3: tanh(x)

Results are written to different output tiles, demonstrating parallel
compute on scattered data.

Grid layout (4x1):
  (0,0) ──scatter──> (0,0), (1,0), (2,0), (3,0)
   │                   │      │      │      │
   │                  abs    neg    exp   tanh
   │                   │      │      │      │
   └─────────────────> out[0] out[1] out[2] out[3]
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@ttl.kernel(grid=(4, 1))
def pipe_scatter_compute(inp, out):
    # Scatter pipe: (0,0) broadcasts to all cores (0,0) through (3,0)
    # This includes loopback since source is in destination range
    scatter_pipe = ttl.Pipe(src=(0, 0), dst=(slice(0, 4), 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)
        with inp_cb.wait() as tile_in:
            with out_cb.reserve() as tile_out:
                # Each core applies a different function
                if x == 0:
                    tile_out.store(ttl.math.abs(tile_in))
                elif x == 1:
                    tile_out.store(ttl.math.neg(tile_in))
                elif x == 2:
                    tile_out.store(ttl.math.exp(tile_in))
                else:
                    tile_out.store(ttl.math.tanh(tile_in))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            # Source core (0,0) reads from DRAM and broadcasts
            with scatter_pipe.if_src():
                tx = ttl.copy(inp[0, 0], inp_blk)
                tx.wait()
                tx = ttl.copy(inp_blk, scatter_pipe)
                tx.wait()

            # All cores (including source) receive the broadcast
            with scatter_pipe.if_dst():
                tx = ttl.copy(scatter_pipe, inp_blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            # Each core writes to its own output tile
            x, y = ttl.core(dims=2)
            tx = ttl.copy(out_blk, out[0, x])
            tx.wait()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @compute
# CHECK: ttl.core

# CHECK-LABEL: func.func @dm_read
# Scatter pipe with loopback (source in destination range)
# CHECK: ttl.create_pipe src(0, 0) dst(0, 0) to(3, 0)
# CHECK: ttl.if_src
# CHECK: ttl.if_dst

# CHECK-LABEL: func.func @dm_write


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        torch.manual_seed(42)
        # Use bounded values for exp/tanh stability
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16) * 0.5
        # Output is 4 tiles wide (one per core)
        out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipe_scatter_compute(inp, out)
        else:
            print("=== Scatter-Compute Test ===")
            print("Scatter: (0,0) -> all cores")
            print("Compute: abs, neg, exp, tanh (one per core)")
            require_hardware()

            pipe_scatter_compute(inp, out)

            out_result = ttnn.to_torch(out)

            # Each output tile should have the corresponding function applied
            tile0 = out_result[:, 0:32]    # abs(x)
            tile1 = out_result[:, 32:64]   # neg(x)
            tile2 = out_result[:, 64:96]   # exp(x)
            tile3 = out_result[:, 96:128]  # tanh(x)

            expected_abs = torch.abs(inp_torch)
            expected_neg = torch.neg(inp_torch)
            expected_exp = torch.exp(inp_torch)
            expected_tanh = torch.tanh(inp_torch)

            print(f"Input sample:        {inp_torch[0, 0:4]}")
            print(f"Output tile 0 (abs): {tile0[0, 0:4]}")
            print(f"Output tile 1 (neg): {tile1[0, 0:4]}")
            print(f"Output tile 2 (exp): {tile2[0, 0:4]}")
            print(f"Output tile 3 (tanh):{tile3[0, 0:4]}")

            assert_allclose(tile0, expected_abs, rtol=0.05, atol=0.05)
            assert_allclose(tile1, expected_neg, rtol=0.05, atol=0.05)
            assert_allclose(tile2, expected_exp, rtol=0.05, atol=0.05)
            assert_allclose(tile3, expected_tanh, rtol=0.05, atol=0.05)

            print("=== Scatter-Compute Test Complete ===")

    finally:
        ttnn.close_device(device)
