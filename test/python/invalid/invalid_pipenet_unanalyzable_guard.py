# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""End-to-end coverage for the `could not statically analyze the PipeNet
guard` diagnostic from `ttl-verify-pipenet-guards`.

The pipe sender is guarded by a predicate read from runtime tensor data. The
verifier cannot prove that the surrounding reserve is restricted to the
PipeNet source nodes, so it rejects the program.
"""

# CHECK: error: could not statically analyze the PipeNet guard

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch  # noqa: E402
import ttnn  # noqa: E402

import ttl  # noqa: E402


def _host_ttnn(shape):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@ttl.operation(grid=(2, 1))
def unanalyzable_guard_pipe(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

    guard_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        x, _ = ttl.node(dims=2)
        if x == 1:
            with inp_cb.wait() as t, out_cb.reserve() as o:
                o.store(t)

    @ttl.datamovement()
    def dm_read():
        node_x, _node_y = ttl.node(dims=2)
        with guard_cb.reserve() as guard_blk:
            ttl.copy(inp[0, 0], guard_blk).wait()

        with guard_cb.wait() as guard_blk:
            runtime_lhs = ttl.raw_element_read(guard_blk, 0, 0)
            runtime_rhs = ttl.raw_element_read(guard_blk, 0, 1)
            runtime_selected = runtime_lhs > runtime_rhs
        coordinate_selected = node_x == 0

        if coordinate_selected != runtime_selected:
            with inp_cb.reserve() as blk:
                ttl.copy(inp[0, 0], blk).wait()

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
        if x == 1:
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, 0]).wait()


def main():
    inp = _host_ttnn((32, 64))
    out = _host_ttnn((32, 32))
    unanalyzable_guard_pipe(inp, out)


if __name__ == "__main__":
    main()
