# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir

"""Ring pipe test: each core sends its tile to the next core, receives from previous.

Each core loads its own tile from DRAM, sends via pipe, receives neighbor via pipe,
adds them together, writes result. This mimics the neighbor-sharing pattern in MD.

Grid layout (4x1):
  Core 0 -> Core 1 -> Core 2 -> Core 3 -> Core 0 (ring)
  Each core: out[x] = inp[x] + inp[(x-1) % 4]
"""

import ttnn
import ttl
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TILE = 32
N_CORES = 4


@ttl.kernel(grid=(N_CORES, 1))
def pipe_ring(inp, out):
    # Forward ring: each core sends to +1, receives from -1 (with wraparound)
    net = ttl.PipeNet([
        ttl.Pipe((x, 0), ((x + 1) % N_CORES, 0))
        for x in range(N_CORES)
    ])

    own_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    nbr_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with own_dfb.wait() as own, nbr_dfb.wait() as nbr, out_dfb.reserve() as o:
            o.store(own + nbr)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.core(dims=2)
        # Load own tile from DRAM
        with own_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, x], blk)
            tx.wait()

            # Send own tile to next core via pipe
            def send(pipe):
                xf = ttl.copy(blk, pipe)
                xf.wait()
            net.if_src(send)

        # Receive neighbor tile from previous core via pipe
        with nbr_dfb.reserve() as blk:
            def recv(pipe):
                xf = ttl.copy(pipe, blk)
                xf.wait()
            net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.core(dims=2)
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0, x])
            tx.wait()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @compute
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_wait
# CHECK: ttl.cb_reserve
# CHECK: ttl.add
# CHECK: ttl.store

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# Ring pipes: 4 unicast pipes (0,0)->(1,0), (1,0)->(2,0), (2,0)->(3,0), (3,0)->(0,0)
# if_src emitted for each pipe
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0)
# CHECK: ttl.if_src
# CHECK: ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0)
# CHECK: ttl.if_src
# CHECK: ttl.create_pipe src(2, 0) dst(3, 0) to(3, 0)
# CHECK: ttl.if_src
# CHECK: ttl.create_pipe src(3, 0) dst(0, 0) to(0, 0)
# CHECK: ttl.if_src

# if_dst emitted for each pipe
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0)
# CHECK: ttl.if_dst
# CHECK: ttl.create_pipe src(1, 0) dst(2, 0) to(2, 0)
# CHECK: ttl.if_dst
# CHECK: ttl.create_pipe src(2, 0) dst(3, 0) to(3, 0)
# CHECK: ttl.if_dst
# CHECK: ttl.create_pipe src(3, 0) dst(0, 0) to(0, 0)
# CHECK: ttl.if_dst

# CHECK-LABEL: func.func @dm_write


if __name__ == "__main__":
    import torch
    from ttlang_test_utils import require_hardware, to_dram, assert_allclose

    compile_only = os.environ.get("TTLANG_COMPILE_ONLY") == "1"

    device = ttnn.open_device(device_id=0)

    try:
        # Input: 4 tiles in a row (1 per core)
        inp_torch = torch.randn(TILE, N_CORES * TILE, dtype=torch.bfloat16)
        out_torch = torch.zeros(TILE, N_CORES * TILE, dtype=torch.bfloat16)

        inp = to_dram(inp_torch, device)
        out = to_dram(out_torch, device)

        if compile_only:
            pipe_ring(inp, out)
        else:
            print("=== Ring Pipe Test ===")
            require_hardware()

            pipe_ring(inp, out)

            result = ttnn.to_torch(out).float()
            inp_f = inp_torch.float()

            # Expected: out[x] = inp[x] + inp[(x-1) % N_CORES]
            all_pass = True
            for x in range(N_CORES):
                own = inp_f[:, x*TILE:(x+1)*TILE]
                nbr = inp_f[:, ((x-1) % N_CORES)*TILE:((x-1) % N_CORES + 1)*TILE]
                expected = own + nbr
                actual = result[:, x*TILE:(x+1)*TILE]
                err = torch.max(torch.abs(actual - expected) / (torch.abs(expected) + 1e-6))
                status = "PASS" if err < 0.01 else "FAIL"
                if status == "FAIL":
                    all_pass = False
                print(f"  Core {x}: max_err={err:.6f} {status}")

            assert all_pass, "Ring pipe test failed"
            print("=== Ring Pipe Test Complete ===")

    finally:
        ttnn.close_device(device)
