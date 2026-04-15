# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
PipeNet unicast: core 1 sends a tile to core 0 via a single unicast pipe.

Grid layout (2x1):
  Core 0: destination (receives via pipe, writes DRAM)
  Core 1: source (reads DRAM, sends via pipe)
"""

import os
import torch
import ttnn
import ttl

os.environ["TTLANG_COMPILE_ONLY"] = "1"


@ttl.operation(grid=(2, 1))
def unicast_pipe(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with inp_cb.wait() as t, out_cb.reserve() as o:
                o.store(t)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        if x == 1:
            with inp_cb.reserve() as blk:
                ttl.copy(inp[0, 1], blk).wait()

                def send(pipe):
                    ttl.copy(blk, pipe).wait()

                net.if_src(send)

        def recv(pipe):
            with inp_cb.reserve() as blk:
                ttl.copy(pipe, blk).wait()

        net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, 0]).wait()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# CHECK: ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0)
# CHECK: ttl.if_src

# CHECK: ttl.create_pipe src(1, 0) dst(0, 0) to(0, 0)
# CHECK: ttl.if_dst

# =============================================================================
# C++ Output Checks (unicast pipe)
# =============================================================================

# Sender side: unicast write + semaphore inc
# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()
# CHECK-CPP: noc_async_write(
# CHECK-CPP: noc_async_write_barrier();
# CHECK-CPP: noc_semaphore_inc(

# Receiver side: wait for sender semaphore, then reset
# CHECK-CPP: experimental::semaphore_wait_min(
# CHECK-CPP: noc_semaphore_set(


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        inp = ttnn.from_torch(
            torch.randn(32, 64, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            torch.zeros(32, 32, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        unicast_pipe(inp, out)
    finally:
        ttnn.close_device(device)
