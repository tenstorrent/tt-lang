# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.initial.mlir %python %s > %t.output 2>&1
# RUN: FileCheck %s < %t.initial.mlir
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
PipeNet scatter: core (0,0) multicasts to cores (1,0)-(3,0) via PipeNet.

Matches the scatter example from the spec:
  net = ttl.PipeNet([ttl.Pipe(src=(x, 0), dst=(x, slice(1, grid_y))) ...])

Grid layout (4x1):
  Core 0: source (reads DRAM, multicasts via pipe)
  Cores 1-3: destinations (receive via pipe, compute abs, write DRAM)
"""

import os
import torch
import ttnn
import ttl

os.environ["TTLANG_COMPILE_ONLY"] = "1"


@ttl.operation(grid=(4, 1))
def scatter(inp, out):
    scatter_net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as tile_in, out_cb.reserve() as tile_out:
            tile_out.store(ttl.math.abs(tile_in))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:

            def read_and_send(pipe):
                tx = ttl.copy(inp[0, 0], blk)
                tx.wait()
                xf = ttl.copy(blk, pipe)
                xf.wait()

            scatter_net.if_src(read_and_send)

            def recv(pipe):
                xf = ttl.copy(pipe, blk)
                xf.wait()

            scatter_net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.node(dims=2)
        with out_cb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[x, y])
            tx.wait()


# =============================================================================
# Initial IR Checks
# =============================================================================

# CHECK-LABEL: func.func @dm_read
# CHECK-SAME: ttl.kernel_thread = #ttkernel.thread<noc>

# PipeNet emits create_pipe + if_src for each pipe in the net
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0)
# CHECK: ttl.if_src

# if_dst call (receive from pipe)
# CHECK: ttl.create_pipe src(0, 0) dst(1, 0) to(3, 0)
# CHECK: ttl.if_dst

# =============================================================================
# C++ Output Checks (multicast pipe)
# =============================================================================

# Multicast sender: wait for receivers ready, write data, signal receivers
# CHECK-CPP: // dm_read
# CHECK-CPP: void kernel_main()
# CHECK-CPP: experimental::semaphore_wait(
# CHECK-CPP: noc_async_write_multicast(
# CHECK-CPP: noc_async_write_barrier();
# CHECK-CPP: noc_semaphore_set_multicast(

# Multicast receiver: reset sem, signal sender ready, wait for data
# CHECK-CPP: noc_semaphore_set(
# CHECK-CPP: noc_semaphore_inc(
# CHECK-CPP: experimental::semaphore_wait(


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0)
    try:
        inp = ttnn.from_torch(
            torch.randn(32, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            torch.zeros(32, 128, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scatter(inp, out)
    finally:
        ttnn.close_device(device)
