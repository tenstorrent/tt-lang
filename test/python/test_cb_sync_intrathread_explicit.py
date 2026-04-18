# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Same pattern as test_cb_sync_intrathread_hang.py but with explicit
.push() / .pop() calls. The sync pass should find the user-written
releases and skip inserting its own, so the emitted C++ matches what
the user wrote. This verifies the structure itself is hardware-sound."""

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
DIM_TILES = 2


def _make_explicit_kernel():
    @ttl.operation(grid="auto")
    def explicit_kernel(inp, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = inp.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        acc_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    # Outer: seed accumulator with first input tile,
                    # then pop so the loop's wait advances to the next
                    # tile (otherwise xj reads the same slot as x0).
                    x0 = inp_dfb.wait()
                    acc0 = acc_dfb.reserve()
                    acc0.store(x0)
                    acc0.push()
                    x0.pop()
                    for _ in range(DIM_TILES - 1):
                        xj = inp_dfb.wait()
                        av = acc_dfb.wait()
                        acc_next = acc_dfb.reserve()
                        acc_next.store(av + xj)
                        acc_next.push()
                        av.pop()
                        xj.pop()
                    final = acc_dfb.wait()
                    o = out_dfb.reserve()
                    o.store(final)
                    o.push()
                    final.pop()

        @ttl.datamovement()
        def dm_read():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    for j in range(DIM_TILES):
                        blk = inp_dfb.reserve()
                        ttl.copy(inp[tile_idx, j], blk).wait()

        @ttl.datamovement()
        def dm_write():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    blk = out_dfb.wait()
                    ttl.copy(blk, out[tile_idx, 0]).wait()

    return explicit_kernel


@pytest.mark.requires_device
def test_intrathread_dfb_explicit_push_pop(device):
    """Same structure as test_cb_sync_intrathread_hang but with user-
    written push/pop. Validates the kernel shape runs on hardware when
    the sync pass has nothing to insert."""
    M, N = TILE, DIM_TILES * TILE
    inp = torch.randn(M, N, dtype=torch.bfloat16)
    out = torch.zeros(M, TILE, dtype=torch.bfloat16)

    golden = inp[:, 0:TILE].float()
    for j in range(1, DIM_TILES):
        golden = golden + inp[:, j * TILE : (j + 1) * TILE].float()

    inp_dev = to_dram(inp, device)
    out_dev = to_dram(out, device)

    kernel = _make_explicit_kernel()
    kernel(inp_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result)
