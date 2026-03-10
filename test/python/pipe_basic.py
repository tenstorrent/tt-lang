# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""Basic pipe test: core 0 sends a tile to core 1 via PipeNet.

Grid layout (2x1):
  Core 0: source (reads from DRAM, sends via pipe)
  Core 1: destination (receives via pipe, writes to DRAM)
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TILE = 32


@ttl.kernel(grid=(2, 1))
def pipe_send_recv(inp, out):
    net = ttl.PipeNet([ttl.Pipe((0, 0), (1, 0))])

    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.core(dims=2)
        if x == 1:
            with inp_dfb.wait() as blk, out_dfb.reserve() as o:
                o.store(blk)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.core(dims=2)
        if x == 0:
            # Core 0: load from DRAM, send via pipe
            with inp_dfb.reserve() as blk:
                tx = ttl.copy(inp[0, 0], blk)
                tx.wait()

                def send(pipe):
                    xf = ttl.copy(blk, pipe)
                    xf.wait()
                net.if_src(send)

        if x == 1:
            # Core 1: receive from pipe
            with inp_dfb.reserve() as blk:
                def recv(pipe):
                    xf = ttl.copy(pipe, blk)
                    xf.wait()
                net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.core(dims=2)
        if x == 1:
            with out_dfb.wait() as blk:
                tx = ttl.copy(blk, out[0, 0])
                tx.wait()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0)
# CHECK: ttl.if_src
# CHECK: ttl.copy
# CHECK: ttl.wait

# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0)
# CHECK: ttl.if_dst
# CHECK: ttl.copy
# CHECK: ttl.wait


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        inp_torch = torch.randn(TILE, TILE, dtype=torch.bfloat16)
        out_torch = torch.zeros(TILE, TILE, dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipe_send_recv(inp, out)
        else:
            print("=== Pipe Basic Test ===")
            require_hardware()

            pipe_send_recv(inp, out)

            result = ttnn.to_torch(out).float()
            expected = inp_torch.float()
            assert_allclose(result, expected, rtol=0.01, atol=0.01)

            print("=== Pipe Basic Test Complete ===")

    finally:
        ttnn.close_device(device)
