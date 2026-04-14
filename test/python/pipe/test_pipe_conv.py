# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pipe forward chain with multi-tile blocks.

3-core pipeline: each core applies a weight to its input tiles and
forwards the last weighted tile to the next core as context. The next
core adds the received context to its first output tile.

Simplified from the Engram pipe_conv_kernel pattern. Tests:
  - Forward pipe chain (PipeNet unicast)
  - Multi-tile blocks (1 x HIDDEN_TILES)
  - Conditional send (only last tile of chunk)
  - Lambda callbacks with PipeNet
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32
N_CORES = 3
HTILES = 2
TILES_PER_CORE = 2


@ttl.operation(grid=(N_CORES, 1))
def pipe_chain(inp, weight, out):
    pipes = [ttl.Pipe(src=(x, 0), dst=(x + 1, 0)) for x in range(N_CORES - 1)]
    net = ttl.PipeNet(pipes)

    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, HTILES), block_count=2)
    w_cb = ttl.make_dataflow_buffer_like(weight, shape=(1, HTILES), block_count=1)
    ctx_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, HTILES), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, HTILES), block_count=2)

    @ttl.compute()
    def compute():
        core_x, _ = ttl.node(dims=2)
        with w_cb.wait() as w:
            for local_t in range(TILES_PER_CORE):
                tile_idx = core_x * TILES_PER_CORE + local_t
                if local_t == 0 and core_x > 0:
                    # First tile on non-first core: add received context
                    with (
                        inp_cb.wait() as x,
                        ctx_cb.wait() as ctx,
                        out_cb.reserve() as o,
                    ):
                        o.store((x + ctx) * w)
                else:
                    with inp_cb.wait() as x, out_cb.reserve() as o:
                        o.store(x * w)

    @ttl.datamovement()
    def dm_read():
        core_x, _ = ttl.node(dims=2)
        # Cores > 0: receive context from previous core
        if core_x > 0:
            with ctx_cb.reserve() as blk:

                def recv(pipe):
                    xf = ttl.copy(pipe, blk)
                    xf.wait()

                net.if_dst(recv)
        # Load weight
        with w_cb.reserve() as blk:
            tx = ttl.copy(weight[0, 0:HTILES], blk)
            tx.wait()
        # Load input tiles
        for local_t in range(TILES_PER_CORE):
            tile_idx = core_x * TILES_PER_CORE + local_t
            with inp_cb.reserve() as blk:
                tx = ttl.copy(inp[tile_idx, 0:HTILES], blk)
                tx.wait()

    @ttl.datamovement()
    def dm_write():
        core_x, _ = ttl.node(dims=2)
        for local_t in range(TILES_PER_CORE):
            tile_idx = core_x * TILES_PER_CORE + local_t
            with out_cb.wait() as blk:
                tx = ttl.copy(blk, out[tile_idx, 0:HTILES])
                tx.wait()
                # Send last tile to next core
                if local_t == TILES_PER_CORE - 1:
                    if core_x < N_CORES - 1:

                        def send(pipe):
                            xf = ttl.copy(blk, pipe)
                            xf.wait()

                        net.if_src(send)


def test_pipe_chain(device):
    """Forward pipe chain with context passing and multi-tile blocks."""
    total_tiles = N_CORES * TILES_PER_CORE
    M = total_tiles * TILE
    N = HTILES * TILE

    torch.manual_seed(42)
    inp_torch = torch.randn(M, N, dtype=torch.bfloat16) * 0.1
    w_torch = torch.full((TILE, N), 2.0, dtype=torch.bfloat16)

    inp_tt = to_dram(inp_torch, device)
    w_tt = to_dram(w_torch, device)
    out_tt = to_dram(torch.zeros(M, N, dtype=torch.bfloat16), device)

    pipe_chain(inp_tt, w_tt, out_tt)

    result = ttnn.to_torch(out_tt)

    # Compute expected: each tile weighted by w, with context from prev core
    expected = torch.zeros(M, N, dtype=torch.bfloat16)
    w_f = w_torch.float()
    inp_f = inp_torch.float()
    last_out = None
    for core in range(N_CORES):
        for local_t in range(TILES_PER_CORE):
            tile_idx = core * TILES_PER_CORE + local_t
            r0, r1 = tile_idx * TILE, (tile_idx + 1) * TILE
            x = inp_f[r0:r1, :]
            if local_t == 0 and core > 0 and last_out is not None:
                tile_out = (x + last_out) * w_f
            else:
                tile_out = x * w_f
            expected[r0:r1, :] = tile_out.to(torch.bfloat16)
            last_out = tile_out

    assert_pcc(expected, result, threshold=0.99)
