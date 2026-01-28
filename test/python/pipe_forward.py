# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""
Pipe forward test - verifies data transfer from core (0,0) to core (1,0).

Core (0,0) reads input tile, sends it to core (1,0) via pipe.
Core (1,0) receives and writes to output.
"""

import ttnn
import ttl


@ttl.kernel(grid=(2, 1))
def pipe_forward(inp, out):
    """Forward data from core 0 to core 1 via pipe."""
    pipe = ttl.Pipe(src=(0, 0), dst=(1, 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        inp_tile = inp_cb.wait()
        out_tile = out_cb.reserve()
        out_tile.store(ttl.math.abs(inp_tile))
        inp_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        with pipe.if_src():
            tx_read = ttl.copy(inp[0, 0], inp_blk)
            tx_read.wait()
            tx_send = ttl.copy(inp_blk, pipe)
            tx_send.wait()

        with pipe.if_dst():
            tx_recv = ttl.copy(pipe, inp_blk)
            tx_recv.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        x, y = ttl.core(dims=2)
        tx_write = ttl.copy(out_blk, out[y, x])
        tx_write.wait()
        out_cb.pop()


# Runtime test kernel - simpler version without pipe multicast
@ttl.kernel(grid=(2, 1))
def pipe_forward_simple(inp, out):
    """Each core reads its own tile (no pipe). Tests multi-core execution."""
    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        inp_tile = inp_cb.wait()
        out_tile = out_cb.reserve()
        out_tile.store(ttl.math.abs(inp_tile))
        inp_cb.pop()
        out_cb.push()

    @ttl.datamovement()
    def dm_read():
        inp_blk = inp_cb.reserve()
        x, y = ttl.core(dims=2)
        tx_read = ttl.copy(inp[y, x], inp_blk)
        tx_read.wait()
        inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        out_blk = out_cb.wait()
        x, y = ttl.core(dims=2)
        tx_write = ttl.copy(out_blk, out[y, x])
        tx_write.wait()
        out_cb.pop()


# =============================================================================
# Initial IR Checks - TTL dialect ops for pipe
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# Check pipe creation
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0)

# Check if_src block
# CHECK: ttl.if_src %{{.+}} : <src(0, 0) dst(1, 0) to(1, 0)>
# CHECK: ttl.copy

# Check if_dst block
# CHECK: ttl.if_dst %{{.+}} : <src(0, 0) dst(1, 0) to(1, 0)>
# CHECK: ttl.copy


if __name__ == "__main__":
    import os
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    # Skip runtime test if in compile-only mode (lit test)
    if os.environ.get("TTLANG_COMPILE_ONLY") == "1":
        print("Compile-only mode, skipping runtime test")
        import sys
        sys.exit(0)

    print("=== Pipe Forward Test ===")
    require_hardware()

    device = ttnn.open_device(device_id=0)

    try:
        # 2x1 grid = 32x64 tensor (2 tiles wide, 1 tile tall)
        inp_torch = torch.full((32, 64), 42.0, dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 64), dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        print("Running pipe forward kernel (simple version for sim)...")
        pipe_forward_simple(inp, out)

        # Verify output - each core processes its own tile
        out_result = ttnn.to_torch(out)
        print(f"Output tensor shape: {out_result.shape}")
        print(f"Output tile 0 (first row, cols 0-4): {out_result[0, :4]}")
        print(f"Output tile 1 (first row, cols 32-36): {out_result[0, 32:36]}")
        expected = torch.full((32, 64), 42.0, dtype=torch.bfloat16)
        assert_allclose(out_result, expected)

        print("=== Pipe Forward Test Complete ===")

    finally:
        ttnn.close_device(device)
