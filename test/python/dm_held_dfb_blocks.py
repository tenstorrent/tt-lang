# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s receives-with-read 2>&1 | FileCheck %s --check-prefix=RECEIVES-WITH-READ
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s first-block-write 2>&1 | FileCheck %s --check-prefix=FIRST-BLOCK-WRITE
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s first-block-read 2>&1 | FileCheck %s --check-prefix=FIRST-BLOCK-READ
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s unused-blocks 2>&1 | FileCheck %s --check-prefix=UNUSED-BLOCKS
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s unused-gather-block 2>&1 | FileCheck %s --check-prefix=UNUSED-GATHER-BLOCK
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s guarded-send-before-next-wait 2>&1 | FileCheck %s --check-prefix=GUARDED-SEND

# RECEIVES-WITH-READ: COMPILED
# FIRST-BLOCK-WRITE: int32_t [[TWO:v[0-9]+]] = 2;
# FIRST-BLOCK-WRITE: .reserve_back([[TWO]]);
# FIRST-BLOCK-WRITE: .push_back([[TWO]]);
# FIRST-BLOCK-WRITE: COMPILED
# FIRST-BLOCK-READ: int32_t [[TWO:v[0-9]+]] = 2;
# FIRST-BLOCK-READ: .wait_front([[TWO]]);
# FIRST-BLOCK-READ: .pop_front([[TWO]]);
# FIRST-BLOCK-READ: COMPILED
# UNUSED-BLOCKS: int32_t [[TWO:v[0-9]+]] = 2;
# UNUSED-BLOCKS: .reserve_back([[TWO]]);
# UNUSED-BLOCKS-NEXT: .push_back([[TWO]]);
# UNUSED-BLOCKS-NOT: push_back
# UNUSED-BLOCKS: COMPILED
# UNUSED-GATHER-BLOCK: int32_t [[THREE:v[0-9]+]] = 3;
# UNUSED-GATHER-BLOCK: .reserve_back([[THREE]]);
# UNUSED-GATHER-BLOCK: .push_back([[THREE]]);
# UNUSED-GATHER-BLOCK-NOT: push_back
# UNUSED-GATHER-BLOCK: COMPILED
# GUARDED-SEND: [[DFB:cb_ctarg_[0-9]+]].wait_front(
# GUARDED-SEND: [[DFB]].reserve_back(
# GUARDED-SEND: [[DFB]].push_back(
# GUARDED-SEND: async_write(CoreLocalMem<uint32_t>([[DFB]].get_read_ptr()), unicast_ep
# GUARDED-SEND: [[DFB]].pop_front(
# GUARDED-SEND: [[DFB]].wait_front(
# GUARDED-SEND: COMPILED

"""Accept data-movement kernels that hold several blocks of one DFB when
`ttl-coalesce-dfb-acquires` merges them into one multi-block acquisition.

The DFB pointer names the first merged block, so any access to it is correct;
a later block is written only by a pipe receive into its own view. The merged
release covers blocks without uses. A read of an earlier published block uses
the other pointer, so it does not address a held block.
"""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target
MODE = sys.argv[1]


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


def make_first_block_write():
    first_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    second_pipe = ttl.Pipe(src=(1, 0), dst=(2, 0))
    first_net = ttl.PipeNet([first_pipe])
    second_net = ttl.PipeNet([second_pipe])

    @ttl.operation(grid=(3, 1))
    def first_block_write(inp, out):
        _first_net = first_net
        _second_net = second_net
        send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

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
                with recv_dfb.reserve() as first, recv_dfb.reserve() as second:
                    ttl.raw_element_write(first, 0, 0, 1.0)
                    second_rx = ttl.copy(second_pipe, second)
                    second_rx.wait()
                with recv_dfb.reserve() as third:
                    ttl.copy(first_pipe, third).wait()

        @ttl.datamovement()
        def writer():
            node_x, _ = ttl.node(dims=2)
            if node_x == 2:
                for column in range(3):
                    with recv_dfb.wait() as received:
                        ttl.copy(received, out[0, column]).wait()

    return first_block_write


def make_first_block_read():
    @ttl.operation(grid=(1, 1))
    def first_block_read(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def reader():
            for column in range(2):
                with dfb.reserve() as blk:
                    ttl.copy(inp[0, column], blk).wait()

        @ttl.datamovement()
        def writer():
            with dfb.wait() as first, dfb.wait() as second:
                value = ttl.raw_element_read(first, 0, 0)
            with result_dfb.reserve() as blk:
                ttl.raw_element_write(blk, 0, 0, value)
            with result_dfb.wait() as blk:
                ttl.copy(blk, out[0, 0]).wait()

    return first_block_read


def make_unused_blocks():
    @ttl.operation(grid=(1, 1))
    def unused_blocks(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def reader():
            first = dfb.reserve()
            second = dfb.reserve()
            first.push()
            second.push()

        @ttl.datamovement()
        def writer():
            with dfb.wait() as blk:
                ttl.copy(blk, out[0, 0]).wait()
            with dfb.wait() as blk:
                ttl.copy(blk, out[0, 1]).wait()

    return unused_blocks


def make_unused_gather_block():
    first_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    second_pipe = ttl.Pipe(src=(1, 0), dst=(2, 0))
    first_net = ttl.PipeNet([first_pipe])
    second_net = ttl.PipeNet([second_pipe])

    @ttl.operation(grid=(3, 1))
    def unused_gather_block(inp, out):
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
                with (
                    recv_dfb.reserve() as first,
                    recv_dfb.reserve() as second,
                    recv_dfb.reserve() as third,
                ):
                    first_rx = ttl.copy(first_pipe, first)
                    second_rx = ttl.copy(second_pipe, second)
                    first_rx.wait()
                    second_rx.wait()

        @ttl.datamovement()
        def writer():
            node_x, _ = ttl.node(dims=2)
            if node_x == 2:
                for column in range(3):
                    with recv_dfb.wait() as received:
                        ttl.copy(received, out[0, column]).wait()

    return unused_gather_block


def make_guarded_send_before_next_wait():
    # A waited block assigned under a node condition is held across a
    # reservation of the same DFB and sent before the next wait; its pop is
    # inserted after the send.
    pipe = ttl.Pipe(src=(0, 0), dst=(1, 0))
    net = ttl.PipeNet([pipe])

    @ttl.operation(grid=(2, 1))
    def guarded_send_before_next_wait(inp, out):
        _net = net
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=3)
        recv_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def reader():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                with dfb.reserve() as staged:
                    ttl.copy(inp[0, 0], staged).wait()
                waited = dfb.wait()
                with dfb.reserve() as reserved:
                    ttl.copy(inp[0, 1], reserved).wait()
                ttl.copy(waited, pipe).wait()
                with dfb.wait() as last:
                    ttl.copy(last, out[0, 1]).wait()
            if node_x == 1:
                with recv_dfb.reserve() as received:
                    ttl.copy(pipe, received).wait()

        @ttl.datamovement()
        def writer():
            node_x, _ = ttl.node(dims=2)
            if node_x == 1:
                with recv_dfb.wait() as received:
                    ttl.copy(received, out[0, 0]).wait()

    return guarded_send_before_next_wait


FACTORIES = {
    "receives-with-read": make_held_reservations_with_read,
    "first-block-write": make_first_block_write,
    "first-block-read": make_first_block_read,
    "unused-blocks": make_unused_blocks,
    "unused-gather-block": make_unused_gather_block,
    "guarded-send-before-next-wait": make_guarded_send_before_next_wait,
}
operation = FACTORIES[MODE]()

tensors = [
    ttnn.from_torch(
        torch.zeros((32, 96), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(2)
]
operation(*tensors)
print("COMPILED")
