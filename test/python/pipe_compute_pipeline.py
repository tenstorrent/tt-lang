# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Compute Pipeline with Per-Stage Operations - demonstrates pipelined compute.

Data flows through 4 cores, each applying a unique transformation:
  Core 0: read input, compute abs, forward result
  Core 1: receive, compute x*x (square), forward result
  Core 2: receive, compute neg, forward result
  Core 3: receive, compute exp, write output

The mathematical result is: exp(neg(square(abs(x)))) = e^(-x^2)
This is a Gaussian-like function where each stage has a lasting effect:
  - abs ensures positive input
  - square amplifies the magnitude
  - neg flips the sign (critical for exp to produce small values)
  - exp produces the final exponential transformation

To avoid CB-to-CB copies, we alternate which CB is input vs output:
  Even cores (0, 2): inp_cb -> compute -> out_cb -> send out_cb
  Odd cores (1, 3):  out_cb -> compute -> inp_cb -> send inp_cb

Grid layout (4x1):
  (0,0) ──pipe01──> (1,0) ──pipe12──> (2,0) ──pipe23──> (3,0)
   abs              square            neg               exp+write
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@ttl.kernel(grid=(4, 1))
def pipe_compute_pipeline(inp, out):
    pipe01 = ttl.Pipe(src=(0, 0), dst=(1, 0))
    pipe12 = ttl.Pipe(src=(1, 0), dst=(2, 0))
    pipe23 = ttl.Pipe(src=(2, 0), dst=(3, 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)
        # Even cores: inp_cb -> out_cb
        # Odd cores: out_cb -> inp_cb
        if x == 0:
            with inp_cb.wait() as tile_in:
                with out_cb.reserve() as tile_out:
                    tile_out.store(ttl.math.abs(tile_in))
        elif x == 1:
            with out_cb.wait() as tile_in:
                with inp_cb.reserve() as tile_out:
                    tile_out.store(tile_in * tile_in)
        elif x == 2:
            with inp_cb.wait() as tile_in:
                with out_cb.reserve() as tile_out:
                    tile_out.store(ttl.math.neg(tile_in))
        else:
            with out_cb.wait() as tile_in:
                with inp_cb.reserve() as tile_out:
                    tile_out.store(ttl.math.exp(tile_in))

    @ttl.datamovement()
    def dm_read():
        # Core 0: Read from DRAM into inp_cb
        with pipe01.if_src():
            with inp_cb.reserve() as inp_blk:
                tx = ttl.copy(inp[0, 0], inp_blk)
                tx.wait()

        # Core 1: Receive from pipe01 into out_cb (sent from core 0's out_cb)
        with pipe01.if_dst():
            with out_cb.reserve() as out_blk:
                tx = ttl.copy(pipe01, out_blk)
                tx.wait()

        # Core 2: Receive from pipe12 into inp_cb (sent from core 1's inp_cb)
        with pipe12.if_dst():
            with inp_cb.reserve() as inp_blk:
                tx = ttl.copy(pipe12, inp_blk)
                tx.wait()

        # Core 3: Receive from pipe23 into out_cb (sent from core 2's out_cb)
        with pipe23.if_dst():
            with out_cb.reserve() as out_blk:
                tx = ttl.copy(pipe23, out_blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        # Core 0: Send from out_cb via pipe01
        with pipe01.if_src():
            with out_cb.wait() as out_blk:
                tx = ttl.copy(out_blk, pipe01)
                tx.wait()

        # Core 1: Send from inp_cb via pipe12
        with pipe12.if_src():
            with inp_cb.wait() as inp_blk:
                tx = ttl.copy(inp_blk, pipe12)
                tx.wait()

        # Core 2: Send from out_cb via pipe23
        with pipe23.if_src():
            with out_cb.wait() as out_blk:
                tx = ttl.copy(out_blk, pipe23)
                tx.wait()

        # Core 3: Write from inp_cb to DRAM
        with pipe23.if_dst():
            with inp_cb.wait() as inp_blk:
                tx = ttl.copy(inp_blk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks - TTL dialect ops for compute pipeline
# =============================================================================

# CHECK-LABEL: func.func @compute
# CHECK: ttl.core

# CHECK-LABEL: func.func @dm_read
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0)
# CHECK: ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0)
# CHECK: ttl.create_pipe src(2, 0) dst(3, 0) to(3, 0)

# CHECK-LABEL: func.func @dm_write


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        torch.manual_seed(42)
        # Use small values to avoid exp overflow after squaring
        # Values in range [-1, 1] so x^2 in [0, 1] and exp(-x^2) in [e^-1, 1]
        inp_torch = torch.randn((32, 32), dtype=torch.bfloat16) * 0.5
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipe_compute_pipeline(inp, out)
        else:
            print("=== 4-Stage Compute Pipeline Test ===")
            print("Pipeline: abs -> square -> neg -> exp")
            print("Expected: exp(-x^2) (Gaussian-like)")
            require_hardware()

            pipe_compute_pipeline(inp, out)

            out_result = ttnn.to_torch(out)

            # exp(neg(square(abs(x)))) = exp(-x^2)
            expected = torch.exp(-inp_torch.abs() ** 2)

            print(f"Input sample:      {inp_torch[0, 0:5]}")
            print(f"Expected exp(-x²): {expected[0, 0:5]}")
            print(f"Output:            {out_result[0, 0:5]}")

            # Show that just abs(x) would be very different
            just_abs = torch.abs(inp_torch)
            print(f"Just abs(x):       {just_abs[0, 0:5]}")

            assert_allclose(out_result, expected, rtol=0.05, atol=0.05)

            print("=== 4-Stage Compute Pipeline Test Complete ===")

    finally:
        ttnn.close_device(device)
