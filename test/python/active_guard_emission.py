# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn, tt-device
# Compile two operations through the Python frontend pipeline and
# FileCheck the final EmitC MLIR for evidence of the active-set guard.
# Catches regressions where ttl-insert-pipenet-active-guards is silently
# dropped from the frontend pipeline string in python/ttl/ttl_api.py —
# a regression that lit tests against `ttlang-opt` alone would miss.
#
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.with_pipenet.mlir TTLANG_OP=with_pipenet %python %s
# RUN: FileCheck %s --input-file=%t.with_pipenet.mlir --check-prefix=WITH-PIPENET
#
# RUN: env TTLANG_COMPILE_ONLY=1 TTLANG_FINAL_MLIR=%t.no_pipenet.mlir TTLANG_OP=no_pipenet %python %s
# RUN: FileCheck %s --input-file=%t.no_pipenet.mlir --check-prefix=NO-PIPENET

"""Frontend-pipeline integration check for the PipeNet active-set guard.

The pass marks each inserted scf.if with a unit attribute
`ttl.pipenet_active_guard`. That attribute does not survive the
scf-to-EmitC lowering, but the structural footprint does: the guard
becomes an `emitc.if` whose predicate is built from `emitc.cmp` and
`emitc.bitwise_or` (the OR across rectangles) on the node coordinates.

An operation that constructs a PipeNet must produce at least one
`emitc.if` in the final MLIR. A straight-line operation that
constructs no PipeNet and contains no user-level conditional must
produce none.
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


@ttl.operation(grid="auto")
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
    device = ttnn.open_device(device_id=0)
    try:
        if op_name == "with_pipenet":
            inp_torch = torch.randn(32, 4 * 32, dtype=torch.bfloat16)
            out_torch = torch.zeros(32, 4 * 32, dtype=torch.bfloat16)
        else:
            inp_torch = torch.randn(32, 32, dtype=torch.bfloat16)
            out_torch = torch.zeros(32, 32, dtype=torch.bfloat16)

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        inp = ttnn.to_memory_config(inp, memory_config=ttnn.L1_MEMORY_CONFIG)
        out = ttnn.to_memory_config(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        if op_name == "with_pipenet":
            with_pipenet_op(inp, out)
        else:
            no_pipenet_op(inp, out)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
