# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 %python %s > %t.output 2>&1
# RUN: FileCheck %s --check-prefix=CHECK-CPP < %t.output

"""
Compile-only test for a DFB accumulator acquired in one conditional region and
updated in a later sibling conditional region under the same predicate.
"""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttnn
import ttl


def _host_ttnn(shape):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


@ttl.operation(grid=(1, 1))
def sibling_conditional_dfb_accumulation(lhs, rhs, out):
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, _ = ttl.node(dims=2)
        route_is_active = node_col == 0
        if route_is_active:
            output_accumulator = out_dfb.reserve()
            output_accumulator.store(
                ttl.block.fill(
                    0,
                    shape=output_accumulator.shape,
                    dtype=output_accumulator.dtype,
                )
            )

        for _ in range(2):
            with lhs_dfb.wait() as lhs_blk, rhs_dfb.wait() as rhs_blk:
                if route_is_active:
                    output_accumulator += lhs_blk @ rhs_blk

        if route_is_active:
            output_accumulator.push()

    @ttl.datamovement()
    def reader():
        for _ in range(2):
            with lhs_dfb.reserve() as lhs_blk:
                ttl.copy(lhs[0:1, 0:1], lhs_blk).wait()
            with rhs_dfb.reserve() as rhs_blk:
                ttl.copy(rhs[0:1, 0:1], rhs_blk).wait()

    @ttl.datamovement()
    def writer():
        node_col, _ = ttl.node(dims=2)
        if node_col == 0:
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:1]).wait()


# CHECK-CPP-LABEL: === compute kernel written
# CHECK-CPP: bool [[ROUTE:v[0-9]+]] =
# CHECK-CPP: if ([[ROUTE]]) {
# CHECK-CPP: reserve_back
# CHECK-CPP: for (size_t
# CHECK-CPP: if ([[ROUTE]]) {
# CHECK-CPP: matmul_block
# CHECK-CPP: pack_tile
# CHECK-CPP: if ([[ROUTE]]) {
# CHECK-CPP: push_back
if __name__ == "__main__":
    lhs = _host_ttnn((32, 32))
    rhs = _host_ttnn((32, 32))
    out = _host_ttnn((32, 32))
    sibling_conditional_dfb_accumulation(lhs, rhs, out)
