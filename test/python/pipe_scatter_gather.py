# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Scatter-Compute-Gather pattern with accumulation.

Core 0 broadcasts input to 3 worker cores, each computes x*x,
then results are gathered back to Core 0 which accumulates them.

Grid layout (4x1):
  Core 0: Scatter source + Gather/Accumulate
  Cores 1-3: Workers (compute x*x)

Flow:
  1. dm_read: Core 0 scatters input to workers via scatter_pipe
  2. compute: Workers compute x*x; Core 0 accumulates from gather_cb
  3. dm_write: Workers send out_cb via gather pipes -> gather_cb on Core 0;
              Core 0 writes out_cb to DRAM

Expected result: 3 * x * x

TODO: Refactor gather pipes to use PipeNet for cleaner multi-source gather pattern.
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@ttl.kernel(grid=(4, 1))
def pipe_scatter_gather(inp, out):
    # Scatter: Core 0 broadcasts to workers 1-3
    scatter_pipe = ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))

    # Gather: Each worker sends back to Core 0
    # TODO: Refactor with PipeNet for cleaner gather pattern
    gather1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    gather2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    gather3 = ttl.Pipe(src=(3, 0), dst=(0, 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)
    gather_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=4)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)
        if x == 0:
            # Core 0: Accumulate 3 gathered results
            with (
                gather_cb.wait() as t1,
                gather_cb.wait() as t2,
                gather_cb.wait() as t3,
                out_cb.reserve() as result,
            ):
                result.store(t1 + t2 + t3)
        else:
            # Workers: compute x*x, write to gather_cb for Core 0 to receive
            with inp_cb.wait() as tile_in, out_cb.reserve() as tile_out:
                tile_out.store(tile_in * tile_in)

    @ttl.datamovement()
    def dm_read():
        # Core 0: Read input and scatter to workers
        with scatter_pipe.if_src():
            with inp_cb.reserve() as inp_blk:
                tx = ttl.copy(inp[0, 0], inp_blk)
                tx.wait()
                tx = ttl.copy(inp_blk, scatter_pipe)
                tx.wait()

        # Workers: Receive scattered data
        with scatter_pipe.if_dst():
            with inp_cb.reserve() as inp_blk:
                tx = ttl.copy(scatter_pipe, inp_blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # Workers: Send out_cb -> gather pipe (arrives at gather_cb on Core 0)
        if x == 1:
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, gather1)
                tx.wait()
        elif x == 2:
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, gather2)
                tx.wait()
        elif x == 3:
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, gather3)
                tx.wait()
        else:
            # Core 0: Receive from all gather pipes into gather_cb
            with gather_cb.reserve() as blk:
                tx1 = ttl.copy(gather1, blk)
            with gather_cb.reserve() as blk:
                tx2 = ttl.copy(gather2, blk)
            with gather_cb.reserve() as blk:
                tx3 = ttl.copy(gather3, blk)
            tx1.wait()
            tx2.wait()
            tx3.wait()
            # Write accumulated result to DRAM
            with out_cb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @compute
# CHECK: ttl.core

# CHECK-LABEL: func.func @dm_read
# Scatter pipe
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0)

# CHECK-LABEL: func.func @dm_write
# Gather pipes
# CHECK: ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0)
# CHECK: ttl.create_pipe src(2, 0) dst(0, 0) to(0, 0)
# CHECK: ttl.create_pipe src(3, 0) dst(0, 0) to(0, 0)


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        torch.manual_seed(42)
        inp_torch = torch.randn((32, 128), dtype=torch.bfloat16) * 0.3
        out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipe_scatter_gather(inp, out)
        else:
            print("=== Scatter-Gather Test ===")
            print("Scatter: Core 0 -> Cores 1-3")
            print("Compute: Each worker does x*x")
            print("Gather: Workers -> Core 0")
            print("Accumulate: Core 0 sums 3 results")
            print("Expected: 3 * x * x")
            require_hardware()

            pipe_scatter_gather(inp, out)

            out_result = ttnn.to_torch(out)

            # Expected: 3 * x^2 (3 workers each computing x*x on same input)
            expected_tile = 3 * (inp_torch[:, 0:32] ** 2)
            actual_tile = out_result[:, 0:32]

            print(f"Input sample:      {inp_torch[0, 0:5]}")
            print(f"Expected 3*x²:     {expected_tile[0, 0:5]}")
            print(f"Output:            {actual_tile[0, 0:5]}")

            assert_allclose(actual_tile, expected_tile, rtol=0.1, atol=0.1)

            print("=== Scatter-Gather Test Complete ===")

    finally:
        ttnn.close_device(device)
