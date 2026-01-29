# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Pipe row multicast test - verifies horizontal multicast within a row.

Core (0,0) broadcasts to cores (1,0), (2,0), (3,0) in the same row.
This tests multicast over the X dimension only.

Grid layout (4x1):
  (0,0) source  -> (1,0) (2,0) (3,0) destinations
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@ttl.kernel(grid=(4, 1))
def pipe_row_multicast(inp, out):
    """Multicast from core (0,0) to cores (1,0), (2,0), (3,0) in same row."""
    # Horizontal multicast: src=(0,0) to cores (1,0) through (3,0)
    pipe = ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))

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
            # Source core (0,0) reads its tile and multicasts horizontally
            with pipe.if_src():
                tx_read = ttl.copy(inp[0, 0], inp_blk)
                tx_read.wait()
                tx_send = ttl.copy(inp_blk, pipe)
                tx_send.wait()

            # Destination cores (1,0), (2,0), (3,0) receive
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
# Initial IR Checks - TTL dialect ops for horizontal multicast pipe
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# Check horizontal multicast pipe (X range, Y fixed)
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0)

# Check if_src block
# CHECK: ttl.if_src %{{.+}} : <src(0, 0) dst(1, 0) to(3, 0)>

# Check if_dst block
# CHECK: ttl.if_dst %{{.+}} : <src(0, 0) dst(1, 0) to(3, 0)>


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        # 4x1 grid = 32x128 tensor (4 tiles wide, 1 tile tall)
        inp_torch = torch.full((32, 128), 42.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 128), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipe_row_multicast(inp, out)
        else:
            print("=== Pipe Row Multicast Test ===")
            require_hardware()

            pipe_row_multicast(inp, out)

            out_result = ttnn.to_torch(out)
            print(f"Output tensor: {out_result[0,:5]}")
            # All cores receive value 42 from source (0,0)
            expected = torch.full((32, 128), 42.0, dtype=torch.bfloat16)
            assert_allclose(out_result, expected)

            print("=== Pipe Row Multicast Test Complete ===")

    finally:
        ttnn.close_device(device)
