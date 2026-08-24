# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Source-level coverage for tile-granular DFB block subviews.

The tests pack and unpack multiple logical payload fields through one DFB block.
This verifies the frontend emits DFB-associated tensor.extract_slice views and
that stores into those subviews preserve the existing DFB lowering contract.
"""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram
from utils.correctness import assert_allclose

TILE = 32


def _three_tile_tensor(values):
    tiles = [torch.full((TILE, TILE), value, dtype=torch.bfloat16) for value in values]
    return torch.cat(tiles, dim=1)


@ttl.operation(grid=(1, 1))
def dfb_subview_pack_unpack(inp, out):
    payload_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 3), block_count=2)

    @ttl.compute()
    def compute():
        with payload_dfb.wait() as payload, out_dfb.reserve() as out_blk:
            out_blk[0:1, 0:1].store(payload[0:1, 2:3])
            out_blk[0:1, 1:2].store(payload[0:1, 0:1])
            out_blk[0:1, 2:3].store(payload[0:1, 1:2])

    @ttl.datamovement()
    def dm_read():
        with payload_dfb.reserve() as payload:
            ttl.copy(inp[0:1, 0:3], payload).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as out_blk:
            ttl.copy(out_blk, out[0:1, 0:3]).wait()


@ttl.operation(grid=(2, 1))
def dfb_subview_pipe_payload(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(1, 0), dst=(0, 0))])
    send_payload_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 3), block_count=2)
    recv_payload_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 3), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 3), block_count=2)

    @ttl.compute()
    def compute():
        if net.is_dst():
            with recv_payload_dfb.wait() as payload, out_dfb.reserve() as out_blk:
                out_blk[0:1, 0:1].store(payload[0:1, 1:2])
                out_blk[0:1, 1:2].store(payload[0:1, 2:3])
                out_blk[0:1, 2:3].store(payload[0:1, 0:1])

    @ttl.datamovement()
    def dm_pipe():
        if net.is_src():
            with send_payload_dfb.reserve() as payload:
                ttl.copy(inp[0:1, 0:3], payload).wait()

        def send(pipe):
            with send_payload_dfb.wait() as payload:
                ttl.copy(payload, pipe).wait()

        net.if_src(send)

        def recv(pipe):
            with recv_payload_dfb.reserve() as payload:
                ttl.copy(pipe, payload).wait()

        net.if_dst(recv)

    @ttl.datamovement()
    def dm_write():
        if net.is_dst():
            with out_dfb.wait() as out_blk:
                ttl.copy(out_blk, out[0:1, 0:3]).wait()


def _run_three_tile_kernel(device, kernel, inp_values, expected_values):
    inp_torch = _three_tile_tensor(inp_values)
    out_torch = torch.zeros(TILE, 3 * TILE, dtype=torch.bfloat16)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out).reshape(TILE, 3 * TILE)
    expected = _three_tile_tensor(expected_values)
    assert_allclose(result.float(), expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.requires_device
def test_dfb_subview_pack_unpack(device):
    _run_three_tile_kernel(
        device, dfb_subview_pack_unpack, [1.0, 2.0, 3.0], [3.0, 1.0, 2.0]
    )


@pytest.mark.requires_device
def test_dfb_subview_pipe_payload(device):
    _run_three_tile_kernel(
        device, dfb_subview_pipe_payload, [4.0, 5.0, 6.0], [5.0, 6.0, 4.0]
    )


def _compile_invalid_subview(device, kernel):
    out = to_dram(torch.zeros(TILE, 3 * TILE, dtype=torch.bfloat16), device)
    kernel(out)


@ttl.operation(grid=(1, 1))
def invalid_dfb_subview_rank(out):
    payload_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 3), block_count=2)

    @ttl.compute()
    def compute():
        with payload_dfb.reserve() as payload:
            payload[0:1].store(ttl.block.fill(1.0, shape=(1, 1)))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def invalid_dfb_subview_oob(out):
    payload_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 3), block_count=2)

    @ttl.compute()
    def compute():
        with payload_dfb.reserve() as payload:
            payload[0:1, 2:4].store(ttl.block.fill(1.0, shape=(1, 2)))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def invalid_dfb_subview_step(out):
    payload_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 3), block_count=2)

    @ttl.compute()
    def compute():
        with payload_dfb.reserve() as payload:
            payload[0:1, 0:3:2].store(ttl.block.fill(1.0, shape=(1, 1)))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@ttl.operation(grid=(1, 1))
def invalid_dfb_subview_store_shape(out):
    payload_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 3), block_count=2)

    @ttl.compute()
    def compute():
        with payload_dfb.reserve() as payload:
            payload[0:1, 0:1].store(ttl.block.fill(1.0, shape=(1, 2)))

    @ttl.datamovement()
    def dm_read():
        pass

    @ttl.datamovement()
    def dm_write():
        pass


@pytest.mark.requires_device
def test_dfb_subview_rank_mismatch_rejected(device):
    with pytest.raises(Exception, match="Expected 2 indices"):
        _compile_invalid_subview(device, invalid_dfb_subview_rank)


@pytest.mark.requires_device
def test_dfb_subview_out_of_bounds_rejected(device):
    with pytest.raises(Exception, match="out of bounds"):
        _compile_invalid_subview(device, invalid_dfb_subview_oob)


@pytest.mark.requires_device
def test_dfb_subview_step_rejected(device):
    with pytest.raises(Exception, match="Slice step is not supported"):
        _compile_invalid_subview(device, invalid_dfb_subview_step)


@pytest.mark.requires_device
def test_dfb_subview_store_shape_mismatch_rejected(device):
    with pytest.raises(Exception, match="must match view shape dimension"):
        _compile_invalid_subview(device, invalid_dfb_subview_store_shape)
