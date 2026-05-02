# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# Frontend-pipeline regression check: catches a dropped
# ttl-insert-pipenet-active-guards in the Python pipeline string.
# Compile-only via TTLANG_COMPILE_ONLY=1; tt-device REQUIRES because
# `ttnn.from_torch(layout=TILE_LAYOUT)` triggers tt-metal cluster init
# even without a device handle (sibling pattern: simple_add.py).
#
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.with_pipenet.mlir TTLANG_OP=with_pipenet %python %s
# RUN: FileCheck %s --input-file=%t.with_pipenet.mlir --check-prefix=WITH-PIPENET
#
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.no_pipenet.mlir TTLANG_OP=no_pipenet %python %s
# RUN: FileCheck %s --input-file=%t.no_pipenet.mlir --check-prefix=NO-PIPENET

"""Frontend-pipeline integration check for the PipeNet active-set guard.

The `ttl.pipenet_active_guard` attribute does not survive scf-to-EmitC,
but the structural footprint does: the guard becomes an `emitc.if`
whose predicate is built from `emitc.cmp` and `emitc.bitwise_or`. A
PipeNet operation must produce one; a straight-line operation must
not.
"""

# WITH-PIPENET: emitc.bitwise_or
# WITH-PIPENET: emitc.if

# NO-PIPENET-NOT: emitc.if
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


# Hardcoded grid: `grid="auto"` resolves via the active device.
@ttl.operation(grid=(8, 7))
def with_pipenet_op(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))])

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as t, out_cb.reserve() as o:
            o.store(ttl.math.abs(t))

    @ttl.datamovement()
    def dm_read():
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
