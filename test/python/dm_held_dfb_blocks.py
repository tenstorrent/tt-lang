# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s 2>&1 | FileCheck %s

"""Accept a data-movement kernel that holds two coalesced reservations of one
DFB, fills each through a pipe receive into its own view, and reads an earlier
published block of the same DFB through the read pointer in the meantime.

The read uses the other pointer, so it does not address either held block.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target


def make_held_reservations_with_read():
    first_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    second_pipe = ttl.Pipe(src=(1, 0), dst=(2, 0))
    first_net = ttl.PipeNet([first_pipe])
    second_net = ttl.PipeNet([second_pipe])

    @ttl.operation(grid=(3, 1))
    def held_reservations_with_read(inp, out):
        _first_net = first_net
        _second_net = second_net
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=3)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def reader():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 0], send_blk).wait()
                    ttl.copy(send_blk, first_pipe).wait()
            if node_x == 1:
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0, 1], send_blk).wait()
                    ttl.copy(send_blk, second_pipe).wait()
            if node_x == 2:
                staged = recv_dfb.reserve()
                ttl.copy(inp[0, 2], staged).wait()
                staged.push()
            if node_x == 2:
                with recv_dfb.reserve() as first, recv_dfb.reserve() as second:
                    first_rx = ttl.copy(first_pipe, first)
                    second_rx = ttl.copy(second_pipe, second)
                    first_rx.wait()
                    second_rx.wait()
                    published = recv_dfb.wait()
                    ttl.copy(published, out[0, 2]).wait()
                    published.pop()
            if node_x == 2:
                for column in range(2):
                    received = recv_dfb.wait()
                    ttl.copy(received, out[0, column]).wait()
                    received.pop()

        @ttl.datamovement()
        def writer():
            pass

    return held_reservations_with_read


tensors = [
    ttnn.from_torch(
        torch.zeros((32, 96), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(2)
]
make_held_reservations_with_read()(*tensors)
print("COMPILED")
# CHECK: COMPILED
