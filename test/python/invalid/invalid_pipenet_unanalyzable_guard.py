# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""End-to-end coverage for the `could not statically analyze the PipeNet
guard` diagnostic from `ttl-verify-pipenet-guards`.

The pipe-coupled `ttl.copy(blk, pipe)` is wrapped in `if (x % 2) == 0`.
The verifier's per-coord folder (`evalIndex` / `evalBool`) handles
add/sub/mul of constants and core coords but not `arith.remsi`, so the
predicate is unanalyzable per launch coord. The verifier rejects the
program and attaches a note to the predicate it couldn't fold; that note
is the artifact `pickEarlierBySourceLoc` chooses deterministically when
two unanalyzable predicates feed the same lattice merge.
"""

# The Python frontend's diagnostic formatter renders the primary error
# and each attached note with its own source-context block.
# CHECK: could not statically analyze the PipeNet guard
# CHECK: this expression is not statically analyzable

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
        x, _ = ttl.node(dims=2)
        # `x % 2` lowers to `arith.remsi`, which the verifier cannot fold
        # per launch coord, so the guard is unanalyzable.
        if (x % 2) == 0:
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
