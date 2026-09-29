# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s guarded 2>&1 | FileCheck %s --check-prefix=GUARDED
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s receive-and-copy 2>&1 | FileCheck %s --check-prefix=RECEIVE-AND-COPY
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s unmerged-shape 2>&1 | FileCheck %s --check-prefix=UNMERGED-SHAPE
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s raw-writes 2>&1 | FileCheck %s --check-prefix=RAW-WRITES
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s held-waits 2>&1 | FileCheck %s --check-prefix=HELD-WAITS
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s held-send 2>&1 | FileCheck %s --check-prefix=HELD-SEND

# GUARDED: error: a data-movement kernel cannot hold two acquired blocks of one dataflow buffer; the earlier block has no use before this acquisition, so use and push it before this acquisition or drop it
# RECEIVE-AND-COPY: error: a data-movement kernel addresses a dataflow buffer through one write pointer, which names the first of the blocks it holds; this operation accesses a later block through it, so push each block before the next acquisition
# UNMERGED-SHAPE: error: a data-movement kernel cannot hold two acquired blocks of one dataflow buffer; this operation accesses the earlier block after the next acquisition returned the same slot, so push the earlier block before that acquisition
# RAW-WRITES: error: a data-movement kernel addresses a dataflow buffer through one write pointer, which names the first of the blocks it holds; this operation accesses a later block through it, so push each block before the next acquisition
# RAW-WRITES: ttl.raw_element_write(second, 0, 0, 2.0)
# HELD-WAITS: error: a data-movement kernel addresses a dataflow buffer through one read pointer, which names the first of the blocks it holds; this operation accesses a later block through it, so pop each block before the next acquisition
# HELD-WAITS: ttl.copy(second, out[0, 1])
# HELD-SEND: error: a data-movement kernel addresses a dataflow buffer through one write pointer, which names the first of the blocks it holds; this operation accesses a later block through it, so push each block before the next acquisition

"""Reject data-movement kernels that hold several blocks of one DFB and
address a later block through the DFB pointer.

A data-movement kernel reaches a DFB through one read or write pointer, so two
acquisitions held together alias unless `ttl-coalesce-dfb-acquires` merges them
into one multi-block acquisition. The pointer then names the first merged
block, so a later block may be written only by a pipe receive into its own
view.
"""

import os
import sys

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn
from ttl import ttl_api

MODE = sys.argv[1]


def _blackhole_compile_target(_runtime_args):
    return "blackhole"


ttl_api._device_target_arch = _blackhole_compile_target


def make_guarded():
    # The blocks are yielded out of the node condition and written through the
    # DFB, with no release before the second acquisition.
    @ttl.operation(grid=(1, 1))
    def guarded(inp, out):
        in_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

        @ttl.datamovement()
        def reader():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                first = in_dfb.reserve()
                second = in_dfb.reserve()
                ttl.copy(inp[0:1, 0:1], first).wait()
                ttl.copy(inp[0:1, 1:2], second).wait()

        @ttl.compute()
        def compute():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                first = in_dfb.wait()
                second = in_dfb.wait()
                result = out_dfb.reserve()
                result.store(first + second)
                first.pop()
                second.pop()
                result.push()

        @ttl.datamovement()
        def writer():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                with out_dfb.wait() as blk:
                    ttl.copy(blk, out[0:1, 0:1]).wait()

    return guarded


def make_gather(receive_second, block_rows):
    first_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    second_pipe = ttl.Pipe(src=(1, 0), dst=(2, 0))
    first_net = ttl.PipeNet([first_pipe])
    second_net = ttl.PipeNet([second_pipe])

    @ttl.operation(grid=(3, 1))
    def gather(inp, out):
        _first_net = first_net
        _second_net = second_net
        send_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(block_rows, block_rows), block_count=2
        )
        recv_dfb = ttl.make_dataflow_buffer_like(
            inp, shape=(block_rows, block_rows), block_count=2
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def reader():
            node_x, _ = ttl.node(dims=2)
            if node_x == 0:
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0:block_rows, 0:block_rows], send_blk).wait()
                    ttl.copy(send_blk, first_pipe).wait()
            if node_x == 1:
                with send_dfb.reserve() as send_blk:
                    ttl.copy(inp[0:block_rows, 0:block_rows], send_blk).wait()
                    ttl.copy(send_blk, second_pipe).wait()
            if node_x == 2:
                with recv_dfb.reserve() as first, recv_dfb.reserve() as second:
                    first_rx = ttl.copy(first_pipe, first)
                    if receive_second:
                        second_rx = ttl.copy(second_pipe, second)
                        second_rx.wait()
                    else:
                        ttl.copy(inp[0:block_rows, 0:block_rows], second).wait()
                    first_rx.wait()

        @ttl.datamovement()
        def writer():
            node_x, _ = ttl.node(dims=2)
            if node_x == 2:
                with recv_dfb.wait() as blk:
                    ttl.copy(blk, out[0:block_rows, 0:block_rows]).wait()
                with recv_dfb.wait() as blk:
                    ttl.copy(blk, out[0:block_rows, 0:block_rows]).wait()

    return gather


def make_raw_writes():
    @ttl.operation(grid=(1, 1))
    def raw_writes(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def reader():
            first = dfb.reserve()
            second = dfb.reserve()
            ttl.raw_element_write(first, 0, 0, 1.0)
            ttl.raw_element_write(second, 0, 0, 2.0)
            first.push()
            second.push()

        @ttl.datamovement()
        def writer():
            with dfb.wait() as blk:
                ttl.copy(blk, out[0, 0]).wait()
            with dfb.wait() as blk:
                ttl.copy(blk, out[0, 1]).wait()

    return raw_writes


def make_held_waits():
    @ttl.operation(grid=(1, 1))
    def held_waits(inp, out):
        dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def reader():
            with dfb.reserve() as blk:
                ttl.copy(inp[0, 0], blk).wait()
            with dfb.reserve() as blk:
                ttl.copy(inp[0, 1], blk).wait()

        @ttl.datamovement()
        def writer():
            with dfb.wait() as first, dfb.wait() as second:
                ttl.copy(first, out[0, 0]).wait()
                ttl.copy(second, out[0, 1]).wait()

    return held_waits


def make_held_send():
    first_pipe = ttl.Pipe(src=(0, 0), dst=(2, 0))
    second_pipe = ttl.Pipe(src=(1, 0), dst=(2, 0))
    forward_pipe = ttl.Pipe(src=(2, 0), dst=(3, 0))
    first_net = ttl.PipeNet([first_pipe])
    second_net = ttl.PipeNet([second_pipe])
    forward_net = ttl.PipeNet([forward_pipe])

    @ttl.operation(grid=(4, 1))
    def held_send(inp, out):
        _first_net = first_net
        _second_net = second_net
        _forward_net = forward_net
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
                    ttl.copy(inp[0, 0], send_blk).wait()
                    ttl.copy(send_blk, second_pipe).wait()
            if node_x == 2:
                # With no wait before it, the send reads through the write
                # pointer, which names the first block.
                with recv_dfb.reserve() as first, recv_dfb.reserve() as second:
                    first_rx = ttl.copy(first_pipe, first)
                    second_rx = ttl.copy(second_pipe, second)
                    first_rx.wait()
                    second_rx.wait()
                    ttl.copy(second, forward_pipe).wait()
            if node_x == 3:
                with recv_dfb.reserve() as blk:
                    ttl.copy(forward_pipe, blk).wait()

        @ttl.datamovement()
        def writer():
            node_x, _ = ttl.node(dims=2)
            if node_x == 2:
                with recv_dfb.wait() as blk:
                    ttl.copy(blk, out[0, 0]).wait()
                with recv_dfb.wait() as blk:
                    ttl.copy(blk, out[0, 1]).wait()
            if node_x == 3:
                with recv_dfb.wait() as blk:
                    ttl.copy(blk, out[0, 0]).wait()

    return held_send


FACTORIES = {
    "guarded": make_guarded,
    # A DRAM read into the second block goes through the DFB pointer.
    "receive-and-copy": lambda: make_gather(receive_second=False, block_rows=1),
    # [2, 2] blocks are not merged, so both receivers would write one slot.
    "unmerged-shape": lambda: make_gather(receive_second=True, block_rows=2),
    # Element writes ignore the view's block offset and land in the first slot.
    "raw-writes": make_raw_writes,
    "held-waits": make_held_waits,
    "held-send": make_held_send,
}
operation = FACTORIES[MODE]()

tensors = [
    ttnn.from_torch(
        torch.zeros((64, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    for _ in range(2)
]
operation(*tensors)
