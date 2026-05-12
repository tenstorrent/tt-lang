# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# Frontend-pipeline integration check for `ttl-verify-pipenet-guards`.
# Compile-only via TTLANG_COMPILE_ONLY=1; tt-device REQUIRES because
# `ttnn.from_torch(layout=TILE_LAYOUT)` triggers tt-metal cluster init
# even without a device handle (sibling pattern: simple_add.py).
#
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_INITIAL_MLIR=%t.with_pipenet_initial.mlir TTLANG_FINAL_MLIR=%t.with_pipenet_final.mlir TTLANG_OP=with_pipenet %python %s
# RUN: FileCheck %s --input-file=%t.with_pipenet_initial.mlir --check-prefix=INITIAL
# RUN: FileCheck %s --input-file=%t.with_pipenet_final.mlir --check-prefix=FINAL
#
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.no_pipenet_final.mlir TTLANG_OP=no_pipenet %python %s
# RUN: FileCheck %s --input-file=%t.no_pipenet_final.mlir --check-prefix=NO-PIPENET

"""Frontend-pipeline integration check for the PipeNet verifier.

`net.is_active()` lowers to `ttl.is_active`, which the verifier
recognizes structurally. After lowering the predicate becomes the
same arith chain the runtime evaluates. A straight-line kernel
without any PipeNet must not contain the predicate machinery.
"""

# Frontend emits the predicate op for `if net.is_active()`.
# INITIAL: ttl.is_active

# After lowering, the user's guard survives as an emitc.if; the
# predicate chain reduces to a bitwise_or over the per-pipe role
# matches.
# FINAL: emitc.bitwise_or
# FINAL: emitc.if

# A kernel without any PipeNet contains neither the emitc.if nor the
# bitwise_or — there is no role predicate to evaluate.
# NO-PIPENET-NOT: emitc.bitwise_or

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


@ttl.operation(grid=(8, 7))
def with_pipenet_op(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        if net.is_active():
            with inp_cb.wait() as t, out_cb.reserve() as o:
                o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
        if net.is_active():
            with inp_cb.reserve() as blk:

                def read_and_send(pipe):
                    ttl.copy(inp[0, 0], blk).wait()
                    ttl.copy(blk, pipe).wait()

                net.if_src(read_and_send)

                def recv(pipe):
                    ttl.copy(pipe, blk).wait()

                net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        if net.is_active():
            x, _ = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, x]).wait()


@ttl.operation(grid=(1, 1))
def no_pipenet_op(inp, out):
    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as t, out_cb.reserve() as o:
            o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:
            ttl.copy(inp[0, 0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_cb.wait() as blk:
            ttl.copy(blk, out[0, 0]).wait()


def main():
    op_name = os.environ.get("TTLANG_OP", "with_pipenet")
    if op_name == "with_pipenet":
        inp = _host_ttnn((32, 4 * 32))
        out = _host_ttnn((32, 4 * 32))
        with_pipenet_op(inp, out)
    else:
        inp = _host_ttnn((32, 32))
        out = _host_ttnn((32, 32))
        no_pipenet_op(inp, out)


if __name__ == "__main__":
    main()
