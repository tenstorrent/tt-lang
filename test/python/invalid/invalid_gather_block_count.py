# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: not %python %s 2>&1 | FileCheck %s

"""
Validation test: gather receiver CB block_count must be >= number of senders.

Each sender writes to a separate slot in the receiver's CB. If block_count is
too small, writes will land outside the CB's allocated memory.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl


@ttl.operation(grid=(3, 1))
def bad_gather(inp, out):
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(1, 0), dst=(0, 0)),
            ttl.Pipe(src=(2, 0), dst=(0, 0)),
        ]
    )

    # block_count=1 is too small: 2 senders need at least 2 blocks
    recv_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with recv_cb.wait() as t, out_cb.reserve() as o:
                o.store(t)

    @ttl.datamovement()
    def dm_read():
        x, _ = ttl.node(dims=2)
        if x > 0:
            with recv_cb.reserve() as blk:
                ttl.copy(inp[0, x], blk).wait()

                def send(pipe):
                    ttl.copy(blk, pipe).wait()

                net.if_src(send)

        def recv(pipe):
            with recv_cb.reserve() as blk:
                ttl.copy(pipe, blk).wait()

        net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        if x == 0:
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, 0]).wait()


# CHECK: gather pipe receiver CB has block_count=1 but 2 senders target it

device = ttnn.open_device(device_id=0)
try:
    inp = ttnn.from_torch(
        torch.randn(32, 96, dtype=torch.bfloat16),
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
    bad_gather(inp, out)
finally:
    ttnn.close_device(device)
