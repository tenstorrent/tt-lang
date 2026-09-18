# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for inserted PipeNet receive completion waits."""

import pytest
import torch

import ttl
from ttlang_test_utils import to_dram
from utils.correctness import assert_pcc

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device

TILE = 32


@ttl.operation(grid=(2, 1))
def receive_wait_runtime_guard(inp, out):
    net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

    guard_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    send_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    receive_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def transfer():
        node_x, _node_y = ttl.node(dims=2)
        with guard_dfb.reserve() as guard_block:
            ttl.copy(inp[0, 0], guard_block).wait()

        with guard_dfb.wait() as guard_block:
            runtime_lhs = ttl.raw_element_read(guard_block, 0, 0)
            runtime_rhs = ttl.raw_element_read(guard_block, 0, 1)
            runtime_selected = runtime_lhs > runtime_rhs
        coordinate_selected = node_x == 1

        def send(pipe):
            with send_dfb.reserve() as send_block:
                ttl.copy(inp[0, 0], send_block).wait()
            with send_dfb.wait() as send_block:
                ttl.copy(send_block, pipe).wait()

        net.if_src(send)

        def receive(pipe):
            with receive_dfb.reserve() as receive_block:
                request = ttl.copy(pipe, receive_block)
                if coordinate_selected != runtime_selected:
                    request.wait()
            with receive_dfb.wait() as receive_block:
                ttl.copy(receive_block, out[0, 0]).wait()

        net.if_dst(receive)

    @ttl.datamovement()
    def second_data_movement():
        pass


@pytest.mark.parametrize(
    "torch_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"]
)
def test_receive_wait_inserted_outside_runtime_guard(device, torch_dtype):
    """Publishing waits even when the conditional source wait does not run."""
    input_torch = torch.zeros((TILE, TILE), dtype=torch_dtype)
    input_torch[0, 0] = 1
    input_tensor = to_dram(input_torch, device)
    output_tensor = to_dram(torch.zeros_like(input_torch), device)

    receive_wait_runtime_guard(input_tensor, output_tensor)

    actual = ttnn.to_torch(output_tensor)
    threshold = 0.999 if torch_dtype == torch.bfloat16 else 0.99999
    assert_pcc(input_torch.float(), actual.float(), threshold=threshold)
