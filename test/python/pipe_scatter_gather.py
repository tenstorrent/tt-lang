# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Pipe scatter-gather test - verifies multicast with loopback (source in destination range).

This tests the scatter-gather pattern where core (0,0) broadcasts to all cores
in its column including itself. This uses multicast with loopback.

Grid layout (1x4):
  (0,0) source AND destination (loopback)
  (0,1) destination
  (0,2) destination
  (0,3) destination
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@ttl.kernel(grid=(1, 4))
def pipe_scatter_gather(inp, out):
    """Broadcast data from core (0,0) to all cores including itself (loopback)."""
    # Multicast with loopback: src=(0,0) is in dst range (0,0) to (0,3)
    pipe = ttl.Pipe(src=(0, 0), dst=(0, slice(0, 4)))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as tile_in:
            with out_cb.reserve() as tile_out:
                tile_out.store(ttl.math.abs(tile_in))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            # Source core (0,0) reads input and multicasts (with loopback)
            with pipe.if_src():
                x, y = ttl.core(dims=2)
                tx_read = ttl.copy(inp[y, x], inp_blk)
                tx_read.wait()
                tx_send = ttl.copy(inp_blk, pipe)
                tx_send.wait()

            # All cores including (0,0) are destinations
            with pipe.if_dst():
                tx_recv = ttl.copy(pipe, inp_blk)
                tx_recv.wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as out_blk:
            x, y = ttl.core(dims=2)
            tx_write = ttl.copy(out_blk, out[y, x])
            tx_write.wait()


# =============================================================================
# Initial IR Checks - TTL dialect ops for loopback multicast pipe
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# Check loopback multicast pipe (source is in destination range)
# CHECK: ttl.create_pipe src(0, 0) dst(0, 0) to(0, 3)

# Check if_src block
# CHECK: ttl.if_src %{{.+}} : <src(0, 0) dst(0, 0) to(0, 3)>

# Check if_dst block (all cores including source)
# CHECK: ttl.if_dst %{{.+}} : <src(0, 0) dst(0, 0) to(0, 3)>


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        # 1x4 grid = 128x32 tensor (1 tile wide, 4 tiles tall)
        inp_torch = torch.full((128, 32), 42.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((128, 32), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipe_scatter_gather(inp, out)
        else:
            print("=== Pipe Scatter-Gather Test ===")
            require_hardware()

            pipe_scatter_gather(inp, out)

            out_result = ttnn.to_torch(out)
            print(f"Output tensor: {out_result[0,:5]}")
            # All cores receive value 42 from source (0,0)
            expected = torch.full((128, 32), 42.0, dtype=torch.bfloat16)
            assert_allclose(out_result, expected)

            print("=== Pipe Scatter-Gather Test Complete ===")

    finally:
        ttnn.close_device(device)
