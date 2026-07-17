# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Direct and composed @ttl.operation with a unicast PipeNet across two cores.

The composed cases use a locally declared PipeNet and an enclosing-scope
PipeNet. Core (1,0) sends a tile over the pipe; core (0,0) adds it to its own
tile and writes the result.

Each pipe buffer is touched on exactly one RISC -- send_cb only in the
if_src callback (BRISC), recv_cb only in if_dst (NCRISC) -- because the
splitter routes the two callbacks onto different RISCs and rejects a
single reserve driven from both."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import assert_allclose, to_dram


@ttl.operation(grid=(2, 1))
def atom_pipe_unicast(a, out):
    own_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    send_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    fwd_net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    node_x, _ = ttl.node(dims=2)

    # Reserves hoisted out of the callbacks; each lands on a single RISC.
    s_blk = send_cb.reserve()
    r_dst = recv_cb.reserve()

    def send(pipe):
        ttl.copy(a[1:2, 0:1], s_blk)
        ttl.copy(s_blk, pipe)

    fwd_net.if_src(send)

    def recv(pipe):
        ttl.copy(pipe, r_dst)

    fwd_net.if_dst(recv)

    if node_x == 0:
        own_blk = own_cb.reserve()
        ttl.copy(a[0:1, 0:1], own_blk)
        s = out_cb.reserve()
        s.store(own_cb.wait() + recv_cb.wait())
        ttl.copy(out_cb.wait(), out[0:1, 0:1])


@ttl.operation()
def _pipe_transfer(a, out, net: ttl.PipeNet):
    own_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    send_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    recv_cb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    out_cb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    node_x, _ = ttl.node(dims=2)

    # Reserves hoisted out of the callbacks; each lands on a single RISC.
    s_blk = send_cb.reserve()
    r_dst = recv_cb.reserve()

    def send(pipe):
        ttl.copy(a[1:2, 0:1], s_blk)
        ttl.copy(s_blk, pipe)

    net.if_src(send)

    def recv(pipe):
        ttl.copy(pipe, r_dst)

    net.if_dst(recv)

    if node_x == 0:
        own_blk = own_cb.reserve()
        ttl.copy(a[0:1, 0:1], own_blk)
        s = out_cb.reserve()
        s.store(own_cb.wait() + recv_cb.wait())
        ttl.copy(out_cb.wait(), out[0:1, 0:1])


@ttl.operation()
def _local_pipe_stage(a, out):
    link = ttl.Pipe(src=(1, 0), dst=(0, 0))
    net = ttl.PipeNet([link])
    _pipe_transfer(a, out, net)


EXTERNAL_NET = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])


@ttl.operation()
def _external_pipe_stage(a, out):
    _pipe_transfer(a, out, EXTERNAL_NET)


@ttl.operation(grid=(2, 1))
def atom_pipe_unicast_local(a, out):
    _local_pipe_stage(a, out)


@ttl.operation(grid=(2, 1))
def atom_pipe_unicast_external(a, out):
    _external_pipe_stage(a, out)


@pytest.mark.parametrize(
    "operation", [atom_pipe_unicast_local, atom_pipe_unicast_external]
)
def test_composed_pipe_unicast(operation, device):
    tile = ttnn.TILE_SIZE
    # Two row-tiles: tile 0 is core (0,0)'s own, tile 1 is sent from (1,0).
    a_t = torch.randn(2 * tile, tile, dtype=torch.bfloat16) * 0.5
    expected = (a_t[:tile].float() + a_t[tile:].float()).to(torch.bfloat16)

    a = to_dram(a_t, device)
    out = to_dram(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    operation(a, out)

    got = ttnn.to_torch(out).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)


def test_atom_pipe_unicast(device):
    tile = ttnn.TILE_SIZE
    # Two row-tiles: tile 0 is core (0,0)'s own, tile 1 is sent from (1,0).
    a_t = torch.randn(2 * tile, tile, dtype=torch.bfloat16) * 0.5
    expected = (a_t[:tile].float() + a_t[tile:].float()).to(torch.bfloat16)

    a = to_dram(a_t, device)
    out = to_dram(torch.zeros(tile, tile, dtype=torch.bfloat16), device)

    atom_pipe_unicast(a, out)

    got = ttnn.to_torch(out).reshape(tile, tile).to(torch.bfloat16)
    assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
