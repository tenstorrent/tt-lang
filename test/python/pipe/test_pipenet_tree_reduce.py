# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for ordered PipeNet tree reduction."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose, assert_pcc  # noqa: E402

pytestmark = pytest.mark.requires_device

CORE_COUNT = 8
TILE_SIZE = 32
ROOT = (0, 0)


def _tree_edges():
    edges = []
    stage_count = (CORE_COUNT - 1).bit_length()
    for stage_index in range(stage_count):
        stride = 1 << stage_index
        edges.extend(
            ((receiver_index + stride, 0), (receiver_index, 0))
            for receiver_index in range(0, CORE_COUNT, 2 * stride)
        )
    return tuple(edges)


def _make_tree_reduce(fp32):
    tree_net = ttl.PipeNet(
        [ttl.Pipe(src=source, dst=destination) for source, destination in _tree_edges()]
    )
    stage_count = (CORE_COUNT - 1).bit_length()
    root_x, root_y = ROOT

    @ttl.operation(
        grid=(CORE_COUNT, 1),
        options=(
            "--ttl-specialize-cores --ttl-accumulation-strategy=l1-pack"
            if fp32
            else "--ttl-specialize-cores"
        ),
        fp32_dest_acc_en=fp32,
    )
    def tree_reduce(source, output):
        staged_input_dfb = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=1
        )
        receive_dfb = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=stage_count
        )
        accumulator_dfb = ttl.make_dataflow_buffer_like(
            source, shape=(1, 1), block_count=1
        )
        output_dfb = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=1)

        @ttl.datamovement()
        def exchange():
            node_x, _node_y = ttl.node(dims=2)
            with staged_input_dfb.reserve() as staged_input_block:
                ttl.copy(source[0, node_x], staged_input_block).wait()

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()

            tree_net.if_dst(receive)

            def send(pipe):
                with accumulator_dfb.wait() as accumulator_block:
                    ttl.copy(accumulator_block, pipe).wait()

            tree_net.if_src(send)

        @ttl.compute()
        def compute():
            node_x, node_y = ttl.node(dims=2)
            destination_count = tree_net.destination_count()
            if node_x == root_x and node_y == root_y:
                with staged_input_dfb.wait() as root_accumulator:
                    for receive_index in range(destination_count):
                        with receive_dfb.wait() as receive_block:
                            root_accumulator = root_accumulator + receive_block
                    with output_dfb.reserve() as output_block:
                        output_block.store_rows(root_accumulator)
            else:
                if tree_net.is_dst():
                    with staged_input_dfb.wait() as node_accumulator:
                        for receive_index in range(destination_count):
                            with receive_dfb.wait() as receive_block:
                                node_accumulator = (
                                    node_accumulator + receive_block
                                )
                        with accumulator_dfb.reserve() as accumulator_block:
                            accumulator_block.store(node_accumulator)
                else:
                    with accumulator_dfb.reserve() as accumulator_block:
                        with staged_input_dfb.wait() as staged_input_block:
                            accumulator_block.store(staged_input_block)

        @ttl.datamovement()
        def store_output():
            node_x, node_y = ttl.node(dims=2)
            if node_x == root_x and node_y == root_y:
                with output_dfb.wait() as output_block:
                    ttl.copy(output_block, output[0, 0]).wait()

    return tree_reduce


TREE_REDUCE_CASES = [
    pytest.param(
        _make_tree_reduce(fp32=False),
        torch.bfloat16,
        5e-2,
        1.0,
        id="bf16",
    ),
    pytest.param(
        _make_tree_reduce(fp32=True),
        torch.float32,
        1e-5,
        1e-5,
        id="fp32",
    ),
]


@pytest.mark.parametrize(("operation", "dtype", "rtol", "atol"), TREE_REDUCE_CASES)
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_tree_reduce(device, operation, dtype, rtol, atol, to_device):
    torch.manual_seed(0)
    source_host = torch.randn(TILE_SIZE, CORE_COUNT * TILE_SIZE, dtype=dtype)
    output_host = torch.zeros(TILE_SIZE, TILE_SIZE, dtype=dtype)
    source = to_device(source_host, device)
    output = to_device(output_host, device)

    operation(source, output)
    ttnn.synchronize_device(device)

    actual = ttnn.to_torch(output).reshape(TILE_SIZE, TILE_SIZE).float()
    expected = (
        torch.stack(
            [
                source_host[
                    :, source_index * TILE_SIZE : (source_index + 1) * TILE_SIZE
                ]
                for source_index in range(CORE_COUNT)
            ]
        )
        .float()
        .sum(dim=0)
    )
    assert_pcc(expected, actual, threshold=0.999)
    assert_allclose(actual, expected, rtol=rtol, atol=atol)

    for tensor in (source, output):
        ttnn.deallocate(tensor)
