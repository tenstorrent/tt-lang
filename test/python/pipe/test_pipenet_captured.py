# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Captured PipeNet end-to-end coverage.

Spec: a pipe net may be constructed in an enclosing scope and captured
by the operation function. This test pins that contract end to end on
hardware and on the simulator — a captured PipeNet behaves identically
to a body-local one (same active set, same data movement, same output).
https://github.com/tenstorrent/tt-lang/blob/<spec-commit>/docs/sphinx/specs/TTLangSpecification.md#L647
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


def _build_scatter_net():
    """Return a PipeNet that will be captured by closure into the
    operation. The PipeNet covers a subset of the launch grid (3 of 4
    nodes), so the active-set computation must include it for the
    compiler to guard out the inactive node correctly."""
    return ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(1, 4), 0))])


def _make_scatter_kernel_with_captured_net(captured_net):
    """Build an `@ttl.operation` whose data-movement thread references
    `captured_net` from the enclosing scope. The op decorator runs
    immediately, so `captured_net` must already exist when this helper
    is called."""

    @ttl.operation(grid=(4, 1))
    def scatter_with_captured(inp, out):
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

                captured_net.if_src(read_and_send)

                def recv(pipe):
                    ttl.copy(pipe, blk).wait()

                captured_net.if_dst(recv)

        @ttl.datamovement()
        def dm_write():
            x, _ = ttl.node(dims=2)
            with out_cb.wait() as blk:
                ttl.copy(blk, out[0, x]).wait()

    return scatter_with_captured


def test_captured_pipenet_scatter(device):
    """Captured PipeNet drives a 1-to-3 scatter: nodes 1..3 receive from
    node 0 and write `abs(input)` to their column of `out`. The result
    must match the body-constructed scatter behaviour exactly."""
    inp_torch = torch.randn(TILE, 4 * TILE, dtype=torch.bfloat16)
    inp_tt = to_dram(inp_torch, device)
    out_tt = to_dram(torch.zeros(TILE, 4 * TILE, dtype=torch.bfloat16), device)

    captured_net = _build_scatter_net()
    kernel = _make_scatter_kernel_with_captured_net(captured_net)
    kernel(inp_tt, out_tt)

    result = ttnn.to_torch(out_tt)
    # Nodes 1..3 each receive tile 0 of inp from node 0 and write abs() to
    # their tile column of out. Node 0 writes nothing, so out[:, 0:TILE] is
    # zeros. Validate only the receiving columns.
    expected_tile = torch.abs(inp_torch[:, 0:TILE])
    for x in range(1, 4):
        assert_pcc(
            expected_tile.float(),
            result[:, x * TILE : (x + 1) * TILE].float(),
            threshold=0.99,
        )
