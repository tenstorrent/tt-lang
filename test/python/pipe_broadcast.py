# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Pipe broadcast test - verifies 2D multicast from one core to entire grid.

Core (0,0) broadcasts to all cores (0,0)-(1,1) using 2D multicast.
This tests multicast over both X and Y dimensions.

Grid layout (2x2):
  (0,0) source and destination
  (1,0) destination
  (0,1) destination
  (1,1) destination
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(2, 2))
def pipe_broadcast(inp, out):
    """Broadcast data from core (0,0) to all cores in 2x2 grid."""
    # 2D multicast: src=(0,0) to all cores (0,0) through (1,1)
    pipe = ttl.Pipe(src=(0, 0), dst=(slice(0, 2), slice(0, 2)))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as tile_in:
            with out_cb.reserve() as tile_out:
                tile_out.store(tile_in)

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as inp_blk:
            # Source core (0,0) reads input and broadcasts to all
            with pipe.if_src():
                tx_read = ttl.copy(inp[0, 0], inp_blk)
                tx_read.wait()
                tx_send = ttl.copy(inp_blk, pipe)
                tx_send.wait()

            # All cores are destinations (including source via loopback)
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
# Initial IR Checks - TTL dialect ops for 2D broadcast pipe
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# Check 2D multicast pipe (both X and Y ranges)
# CHECK: ttl.create_pipe src(0, 0) dst(0, 0) to(1, 1)

# Check if_src block
# CHECK: ttl.if_src %{{.+}} : <src(0, 0) dst(0, 0) to(1, 1)>

# Check if_dst block
# CHECK: ttl.if_dst %{{.+}} : <src(0, 0) dst(0, 0) to(1, 1)>


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Pipe Broadcast Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        # 2x2 grid = 64x64 tensor (2 tiles x 2 tiles)
        inp_torch = torch.full((64, 64), 42.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((64, 64), dtype=torch.bfloat16)

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

        print("Running pipe broadcast kernel...")
        pipe_broadcast(inp, out)

        print("=== Pipe Broadcast Test Complete ===")

    finally:
        ttnn.close_device(device)
