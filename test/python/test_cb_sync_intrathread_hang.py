# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproducer: TTLInsertCBSync places cb_push for an intra-thread
DFB after an scf.for whose body contains a cb_wait on the same DFB.
The first loop iteration's cb_wait blocks forever because the outer
reserve's push has not been emitted yet.

Pattern that triggers the bug:

    tmp = tmp_dfb.reserve()
    tmp.store(<value>)
    # compiler should emit ttl.cb_push on tmp_dfb here
    for _ in range(K):
        y = tmp_dfb.wait()       # <-- hangs on iter 0; push is after loop
        o = out_dfb.reserve()
        o.store(y)

Expected post-fix behavior: cb_push on the outer reserve is placed before
the scf.for, not after it, so the first iteration's wait sees the data.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v --tb=short

import os
import sys

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_pcc  # noqa: E402

TILE = 32


def _make_hang_kernel():
    @ttl.operation(grid="auto")
    def hang_kernel(inp, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = inp.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        # Intra-thread DFB: compute reserves + stores OUTSIDE the loop,
        # the loop body waits on the same DFB. Without the fix, the sync
        # pass places cb_push after scf.for and the wait inside deadlocks.
        tmp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Outer: one reserve + store into tmp_dfb.
                    x = inp_dfb.wait()
                    tmp = tmp_dfb.reserve()
                    tmp.store(x)
                    # Loop body reads tmp_dfb — this is the nested-wait
                    # pattern that the fix must handle.
                    for _ in range(1):
                        y = tmp_dfb.wait()
                        o = out_dfb.reserve()
                        o.store(y)

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    blk = inp_dfb.reserve()
                    ttl.copy(inp[tile_idx, 0], blk).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    blk = out_dfb.wait()
                    ttl.copy(blk, out[tile_idx, 0]).wait()

    return hang_kernel


@pytest.mark.requires_device
def test_intrathread_dfb_outer_reserve_then_loop_wait(device):
    """Without the TTLInsertCBSync fix, cb_push for tmp_dfb is placed
    after the scf.for loop; the wait inside the loop body then blocks
    forever. With the fix, the push precedes the loop and the value
    passes through to the output unchanged."""
    M = TILE
    inp = torch.randn(M, TILE, dtype=torch.bfloat16)
    out = torch.zeros(M, TILE, dtype=torch.bfloat16)

    golden = inp.float()

    inp_dev = to_dram(inp, device)
    out_dev = to_dram(out, device)

    kernel = _make_hang_kernel()
    kernel(inp_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result)
