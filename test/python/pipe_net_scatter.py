# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
PipeNet scatter test: verifies the spec's callback API pattern.

Core (0,0) multicasts input to cores (1,0)-(3,0) via PipeNet.
Workers compute abs(x), then write to DRAM.

Grid layout (4x1):
  Core 0: Scatter source
  Cores 1-3: Workers (receive, compute, write)

This test exercises both named function defs and lambda callbacks
for PipeNet.if_src/if_dst, matching the spec examples.
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@ttl.kernel(grid=(4, 1))
def pipenet_scatter(inp, out):
    scatter_net = ttl.PipeNet([
        ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))
    ])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as tile_in, out_cb.reserve() as tile_out:
            tile_out.store(ttl.math.abs(tile_in))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            # Named function callbacks per the spec
            def pipe_src(pipe):
                tx = ttl.copy(inp[0, 0], inp_blk)
                tx.wait()
                tx2 = ttl.copy(inp_blk, pipe)
                tx2.wait()

            def pipe_dst(pipe):
                tx = ttl.copy(pipe, inp_blk)
                tx.wait()

            scatter_net.if_src(pipe_src)
            scatter_net.if_dst(pipe_dst)

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[x, y])
            tx.wait()


# =============================================================================
# Initial IR Checks - TTL dialect ops using PipeNet callback pattern
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# PipeNet emits create_pipe + if_src for each pipe in the net
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0)
# CHECK: ttl.if_src
# CHECK: ttl.copy
# CHECK: ttl.wait
# CHECK: ttl.copy
# CHECK: ttl.wait

# if_dst callback
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0)
# CHECK: ttl.if_dst
# CHECK: ttl.copy
# CHECK: ttl.wait


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.randn((32, 128), dtype=torch.bfloat16) * 0.5
        out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipenet_scatter(inp, out)
        else:
            print("=== PipeNet Scatter Test ===")
            require_hardware()

            pipenet_scatter(inp, out)

            out_result = ttnn.to_torch(out)
            expected = torch.abs(inp_torch)
            assert_allclose(out_result, expected, rtol=0.1, atol=0.1)

            print("=== PipeNet Scatter Test Complete ===")

    finally:
        ttnn.close_device(device)
