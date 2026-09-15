# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Constant-initialized node membership guards preserve DFB ownership."""

import pytest
import torch
import ttl

from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device

TILE = 32
ITERATIONS = 64


def make_guarded_transfer():
    @ttl.operation(grid=(3, 1))
    def guarded_transfer(input_tensor, output_tensor):
        pipe_net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
        send_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=1
        )
        receive_dfb = ttl.make_dataflow_buffer_like(
            input_tensor, shape=(1, 1), block_count=1
        )

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def transfer():
            for iteration in range(ITERATIONS):

                def send(pipe):
                    with send_dfb.reserve() as send_block:
                        ttl.copy(input_tensor[iteration, 0], send_block).wait()
                    with send_dfb.wait() as send_block:
                        ttl.copy(send_block, pipe).wait()

                pipe_net.if_src(send)

                def receive(pipe):
                    with receive_dfb.reserve() as receive_block:
                        ttl.copy(pipe, receive_block).wait()

                pipe_net.if_dst(receive)

        @ttl.datamovement()
        def store_result():
            node_x, _node_y = ttl.node(dims=2)
            selected = False
            selected = selected or node_x == 1
            if selected:
                for iteration in range(ITERATIONS):
                    with receive_dfb.wait() as receive_block:
                        ttl.copy(receive_block, output_tensor[iteration, 0]).wait()

    return guarded_transfer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_pipenet_boolean_guard(device, dtype, to_device):
    # Distinct tiles expose missing iterations or premature DFB storage reuse.
    input_host = torch.randn(ITERATIONS * TILE, TILE, dtype=dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    make_guarded_transfer()(input_tensor, output_tensor)

    assert_allclose(
        input_host.float(), ttnn.to_torch(output_tensor).float(), rtol=0, atol=0
    )
