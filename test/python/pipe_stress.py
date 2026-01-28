# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Pipe stress test - verifies large-scale multicast on 8x8 grid.

Core (0,0) broadcasts to all 64 cores in an 8x8 grid using 2D multicast.
This is a stress test for the pipe lowering with maximum multicast range.

Grid layout (8x8):
  (0,0) source and destination (loopback)
  All 64 cores are destinations
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import ttnn
import ttl


@ttl.kernel(grid=(8, 8))
def pipe_stress(inp, out):
    """Broadcast data from core (0,0) to all 64 cores in 8x8 grid."""
    # Full 2D multicast: src=(0,0) to all cores (0,0) through (7,7)
    pipe = ttl.Pipe(src=(0, 0), dst=(slice(0, 8), slice(0, 8)))

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
            # Source core (0,0) reads input and broadcasts to all 64 cores
            with pipe.if_src():
                tx_read = ttl.copy(inp[0, 0], inp_blk)
                tx_read.wait()
                tx_send = ttl.copy(inp_blk, pipe)
                tx_send.wait()

            # All 64 cores are destinations (including source via loopback)
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
# Initial IR Checks - TTL dialect ops for 8x8 broadcast pipe
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# Check full 8x8 multicast pipe
# CHECK: ttl.create_pipe src(0, 0) dst(0, 0) to(7, 7)

# Check if_src block
# CHECK: ttl.if_src %{{.+}} : <src(0, 0) dst(0, 0) to(7, 7)>

# Check if_dst block
# CHECK: ttl.if_dst %{{.+}} : <src(0, 0) dst(0, 0) to(7, 7)>


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware

    print("=== Pipe Stress Test (8x8 Grid) ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        # 8x8 grid = 256x256 tensor (8 tiles x 8 tiles)
        inp_torch = torch.full((256, 256), 42.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((256, 256), dtype=torch.bfloat16)

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

        print("Running pipe stress kernel (8x8 broadcast)...")
        pipe_stress(inp, out)

        print("=== Pipe Stress Test Complete ===")

    finally:
        ttnn.close_device(device)
