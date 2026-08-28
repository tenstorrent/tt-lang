# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for ordered PipeNet tree reduction."""

import pytest
import torch

import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram  # noqa: E402
from utils.correctness import assert_allclose, assert_pcc  # noqa: E402

pytestmark = pytest.mark.requires_device

CORE_COUNT = 8
VALID_ROWS = 14
TILE_WIDTH = 32
COMPACT_TILE = (1, TILE_WIDTH)
FULL_TILE = (32, TILE_WIDTH)
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


def _make_tree_reduce(data_format, bytes_per_element):
    tree_net = ttl.PipeNet(
        [
            ttl.Pipe(src=source, dst=destination)
            for source, destination in _tree_edges()
        ]
    )
    payload_size_bytes = VALID_ROWS * TILE_WIDTH * bytes_per_element
    stage_count = (CORE_COUNT - 1).bit_length()

    @ttl.operation(grid=(CORE_COUNT, 1), options="--ttl-specialize-cores")
    def tree_reduce(source, output):
        source_dfb = ttl.make_tensor_backed_dfb(
            source, shape=(1, VALID_ROWS), block_count=1
        )
        staged_input_dfb = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=FULL_TILE
        )
        receive_dfb = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=stage_count,
            tile=FULL_TILE,
        )
        accumulator_dfb = ttl.make_dfb(
            data_format, shape=(1, 1), block_count=1, tile=FULL_TILE
        )
        output_dfb = ttl.make_tensor_backed_dfb(
            output, shape=(1, VALID_ROWS), block_count=1
        )

        @ttl.datamovement()
        def stage_input():
            source_dfb.publish()
            with source_dfb.wait() as source_block:
                with staged_input_dfb.reserve() as staged_input_block:
                    ttl.copy(
                        source_block,
                        staged_input_block,
                        byte_count=payload_size_bytes,
                    ).wait()

        @ttl.datamovement()
        def exchange():
            node_x, node_y = ttl.node(dims=2)

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(
                        pipe,
                        receive_block,
                        byte_count=payload_size_bytes,
                    ).wait()

            tree_net.if_dst(receive)

            def send(pipe):
                with accumulator_dfb.wait() as accumulator_block:
                    ttl.copy(
                        accumulator_block,
                        pipe,
                        byte_count=payload_size_bytes,
                    ).wait()

            tree_net.if_src(send)

            if node_x == ROOT[0] and node_y == ROOT[1]:
                with accumulator_dfb.wait() as accumulator_block:
                    with output_dfb.reserve() as output_block:
                        ttl.copy(
                            accumulator_block,
                            output_block,
                            byte_count=payload_size_bytes,
                        ).wait()

        @ttl.compute()
        def compute():
            with accumulator_dfb.reserve() as accumulator_block:
                with staged_input_dfb.wait() as staged_input_block:
                    accumulator_block.store(staged_input_block)
                if tree_net.is_dst():
                    for receive_index in range(tree_net.destination_count()):
                        with receive_dfb.wait() as receive_block:
                            accumulator_block += receive_block

    return tree_reduce


def _core_ranges(nodes):
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(node_x, node_y),
                ttnn.CoreCoord(node_x, node_y),
            )
            for node_x, node_y in nodes
        }
    )


def _height_sharded_memory_config(nodes):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            _core_ranges(nodes),
            (VALID_ROWS, TILE_WIDTH),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


TREE_REDUCE_CASES = [
    pytest.param(
        _make_tree_reduce("bf16", 2),
        torch.bfloat16,
        5e-2,
        1.0,
        id="bf16",
    ),
    pytest.param(
        _make_tree_reduce("float32", 4),
        torch.float32,
        1e-5,
        1e-5,
        id="fp32",
    ),
]


@pytest.mark.parametrize(
    ("operation", "dtype", "rtol", "atol"), TREE_REDUCE_CASES
)
def test_tree_reduce(device, operation, dtype, rtol, atol):
    torch.manual_seed(0)
    source_host = torch.randn(
        CORE_COUNT * VALID_ROWS, TILE_WIDTH, dtype=dtype
    )
    output_host = torch.zeros(VALID_ROWS, TILE_WIDTH, dtype=dtype)

    source_dram = to_dram(source_host, device, tile=COMPACT_TILE)
    output_dram = to_dram(output_host, device, tile=COMPACT_TILE)
    source = ttnn.to_memory_config(
        source_dram,
        _height_sharded_memory_config(
            tuple((node_x, 0) for node_x in range(CORE_COUNT))
        ),
    )
    output = ttnn.to_memory_config(
        output_dram, _height_sharded_memory_config((ROOT,))
    )

    operation(source, output)
    ttnn.synchronize_device(device)

    actual = ttnn.to_torch(output).reshape(VALID_ROWS, TILE_WIDTH).float()
    expected = (
        source_host.reshape(CORE_COUNT, VALID_ROWS, TILE_WIDTH)
        .float()
        .sum(dim=0)
    )
    assert_pcc(expected, actual, threshold=0.999)
    assert_allclose(actual, expected, rtol=rtol, atol=atol)

    for tensor in (source, output, source_dram, output_dram):
        ttnn.deallocate(tensor)
