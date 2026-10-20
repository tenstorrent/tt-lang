# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: env TTLANG_COMPILE_ONLY=1 not %python %s 2>&1 | FileCheck %s

"""Reject a waited block held across a loop that waits on the same DFB."""

import os

os.environ["TTLANG_COMPILE_ONLY"] = "1"

import torch
import ttl
import ttnn

N_ITERS = 3


@ttl.operation(grid=(1, 1))
def dfb_wait_held_across_loop(a_seed, delta, out):
    a_cb = ttl.make_dataflow_buffer_like(a_seed, shape=(1, 1), block_count=2)
    delta_cb = ttl.make_dataflow_buffer_like(
        delta, shape=(1, 1), block_count=N_ITERS + 1
    )
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        # CHECK: error: dataflow buffer block is still acquired when a nested region acquires the same buffer again; release it before that region (pop it) or acquire the block inside the region
        with a_cb.wait() as a, delta_cb.wait() as d_init:
            acc = a + d_init
            for _ in range(N_ITERS):
                with delta_cb.wait() as d:
                    acc += d
            with out_cb.reserve() as o:
                o.store(acc)

    @ttl.datamovement()
    def reader():
        with a_cb.reserve() as blk:
            ttl.copy(a_seed[0:1, 0:1], blk).wait()
        for _ in range(N_ITERS + 1):
            with delta_cb.reserve() as blk:
                ttl.copy(delta[0:1, 0:1], blk).wait()

    @ttl.datamovement()
    def writer():
        with out_cb.wait() as blk:
            ttl.copy(blk, out[0:1, 0:1]).wait()


if __name__ == "__main__":
    a_seed = ttnn.from_torch(
        torch.ones((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    delta = ttnn.from_torch(
        torch.ones((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    out = ttnn.from_torch(
        torch.zeros((32, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    dfb_wait_held_across_loop(a_seed, delta, out)
