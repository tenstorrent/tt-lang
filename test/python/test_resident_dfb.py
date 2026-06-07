# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Resident DFB primitives (read()/store()): a compute-thread-local
accumulator written via store() and read in place via read(). No CB handshake
is emitted for the accumulator -- reserve_back once-per-pack with no push_back
(write pointer fixed) and no wait_front/pop_front (read pointer fixed) -- so the
slot is packed and read in place. Mirrors test_cb_sync_intrathread_explicit.py
but the accumulator uses the resident primitives instead of wait/reserve/push/pop.
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
DIM_TILES = 4


def _make_resident_kernel():
    @ttl.operation(grid="full")
    def resident_kernel(inp, out):
        grid_cols, _ = ttl.grid_size(dims=2)
        seq_tiles = inp.shape[0] // TILE
        tiles_per_core = -(-seq_tiles // grid_cols)

        inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)
        # Resident accumulator: produced AND consumed in compute via store()/
        # read(), so it lowers with no CB handshake.
        acc_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=1)

        @ttl.compute()
        def compute():
            core_x, _ = ttl.node(dims=2)
            for local_t in range(tiles_per_core):
                tile_idx = core_x * tiles_per_core + local_t
                if tile_idx < seq_tiles:
                    x0 = inp_dfb.wait()
                    acc_dfb.store(x0)  # resident pack, in place
                    x0.pop()
                    for _ in range(DIM_TILES - 1):
                        xj = inp_dfb.wait()
                        av = acc_dfb.read()  # resident read, in place
                        acc_dfb.store(av + xj)
                        xj.pop()
                    final = acc_dfb.read()
                    o = out_dfb.reserve()
                    o.store(final)
                    o.push()

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

    return resident_kernel


@pytest.mark.requires_device
def test_resident_accumulator(device):
    """Sum DIM_TILES input tiles into a resident accumulator and check PCC."""
    M, N = TILE, DIM_TILES * TILE
    inp = torch.randn(M, N, dtype=torch.bfloat16)
    out = torch.zeros(M, TILE, dtype=torch.bfloat16)

    golden = inp[:, 0:TILE].float()
    for j in range(1, DIM_TILES):
        golden = golden + inp[:, j * TILE : (j + 1) * TILE].float()

    inp_dev = to_dram(inp, device)
    out_dev = to_dram(out, device)

    kernel = _make_resident_kernel()
    kernel(inp_dev, out_dev)

    result = ttnn.to_torch(out_dev).float()
    assert_pcc(golden, result)
