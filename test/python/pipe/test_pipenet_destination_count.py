# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for PipeNet destination-record multiplicity."""

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose

pytestmark = pytest.mark.requires_device

TILE = 32
GRID_X = 4
MAX_RECEIVES = 3


# The receive loop uses the same record multiplicity as the PipeNet callbacks.
@ttl.operation(grid=(GRID_X, 1))
def destination_count_gather(inp, out):
    net = ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(3, 0)),
            ttl.Pipe(src=(1, 0), dst=(3, 0)),
            ttl.Pipe(src=(2, 0), dst=(3, 0)),
            ttl.Pipe(src=(0, 0), dst=(2, 0)),
        ]
    )
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    receive_dfb = ttl.make_dataflow_buffer_like(
        out, shape=(1, 1), block_count=MAX_RECEIVES
    )

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm():
        node_x, _node_y = ttl.node(dims=2)

        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                ttl.copy(pipe, receive_block).wait()

        net.if_dst(receive)

        def send(pipe):
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, node_x], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        net.if_src(send)

        for receive_index in range(net.destination_count()):
            with receive_dfb.wait() as receive_block:
                ttl.copy(receive_block, out[receive_index, node_x]).wait()

    @ttl.datamovement()
    def dm_brisc():
        pass


# Destination counts must bound every receive across supported tensor storage.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_destination_count_matches_receive_records(device, dtype, to_device):
    torch.manual_seed(0)
    inp_torch = torch.randn(TILE, GRID_X * TILE, dtype=dtype)
    input_tensor = to_device(inp_torch, device)
    output = to_device(
        torch.zeros(MAX_RECEIVES * TILE, GRID_X * TILE, dtype=dtype), device
    )

    destination_count_gather(input_tensor, output)
    ttnn.synchronize_device(device)

    expected = torch.zeros(MAX_RECEIVES * TILE, GRID_X * TILE, dtype=dtype)
    expected[0:TILE, 2 * TILE : 3 * TILE] = inp_torch[:, 0:TILE]
    for receive_index, source_x in enumerate((0, 1, 2)):
        expected[
            receive_index * TILE : (receive_index + 1) * TILE,
            3 * TILE : 4 * TILE,
        ] = inp_torch[:, source_x * TILE : (source_x + 1) * TILE]

    assert_allclose(expected.float(), ttnn.to_torch(output).float(), rtol=0.0, atol=0.0)
