# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Module-scope PipeNet end-to-end coverage.

A `ttl.PipeNet` constructed at module top-level (the module is an
enclosing scope of the @ttl.operation function per spec L647) must
behave identically to a body-local or closure-captured net. Tests
hardware and simulator.

https://github.com/tenstorrent/tt-lang/blob/<spec-commit>/docs/sphinx/specs/TTLangSpecification.md#L647
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_pcc, to_dram

TILE = 32

# Module-scope PipeNet referenced from the operation's data-movement
# thread by name.
MODULE_SCATTER_NET = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))])


@ttl.operation(grid=(4, 1))
def scatter_with_module_net(inp, out):
    inp_cb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with inp_cb.wait() as tile_in, out_cb.reserve() as tile_out:
            tile_out.store(ttl.math.abs(tile_in))

    @ttl.datamovement()
    def dm_read():
        with inp_cb.reserve() as blk:

            def read_and_send(pipe):
                ttl.copy(inp[0, 0], blk).wait()
                ttl.copy(blk, pipe).wait()

            MODULE_SCATTER_NET.if_src(read_and_send)

            def recv(pipe):
                ttl.copy(pipe, blk).wait()

            MODULE_SCATTER_NET.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        x, _ = ttl.node(dims=2)
        with out_cb.wait() as blk:
            ttl.copy(blk, out[0, x]).wait()


def test_module_scope_pipenet_scatter(device):
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 4 * TILE, dtype=torch.bfloat16), device)

    scatter_with_module_net(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    expected_tile = torch.abs(inp_torch[:, 0:TILE])
    for x in range(1, 4):
        assert_pcc(expected_tile.float(), result[:, x * TILE : (x + 1) * TILE].float())
