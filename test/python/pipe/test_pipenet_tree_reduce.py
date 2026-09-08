# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for ordered PipeNet tree reduction."""

import pytest
import torch

import ttl
import ttl.ttl_api as ttl_api

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose, assert_pcc  # noqa: E402

pytestmark = pytest.mark.requires_device

CORE_COUNT = 8
TILE_SIZE = 32
OUTPUT_ROWS = 14
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


def _make_tree_reduce(fp32, specialize_cores):
    tree_net = ttl.PipeNet(
        [ttl.Pipe(src=source, dst=destination) for source, destination in _tree_edges()]
    )
    stage_count = (CORE_COUNT - 1).bit_length()
    root_x, root_y = ROOT

    options = []
    if specialize_cores:
        options.append("--ttl-specialize-cores")
    if fp32:
        options.append("--ttl-accumulation-strategy=l1-pack")

    @ttl.operation(
        grid=(CORE_COUNT, 1),
        options=" ".join(options),
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
                with output_dfb.reserve() as output_block:
                    with staged_input_dfb.wait() as staged_input_block:
                        output_block.store_rows(staged_input_block)
                    for receive_index in range(destination_count):
                        with receive_dfb.wait() as receive_block:
                            output_block.accumulate_rows(receive_block)
            else:
                with accumulator_dfb.reserve() as accumulator_block:
                    with staged_input_dfb.wait() as staged_input_block:
                        accumulator_block.store(staged_input_block)
                    if tree_net.is_dst():
                        for receive_index in range(destination_count):
                            with receive_dfb.wait() as receive_block:
                                accumulator_block += receive_block

        @ttl.datamovement()
        def store_output():
            node_x, node_y = ttl.node(dims=2)
            if node_x == root_x and node_y == root_y:
                with output_dfb.wait() as output_block:
                    ttl.copy(output_block, output[0:1, 0:OUTPUT_ROWS]).wait()

    return tree_reduce


TREE_REDUCE_CASES = [
    pytest.param(
        torch.bfloat16,
        5e-2,
        1.0,
        id="bf16",
    ),
    pytest.param(
        torch.float32,
        1e-5,
        1e-5,
        id="fp32",
    ),
]


def _record_pipelines(monkeypatch):
    """Record expanded Python compiler pipelines without changing their execution."""
    pipelines = []
    parse_pipeline = ttl_api.PassManager.parse

    class RecordingPassManager:
        @staticmethod
        def parse(pipeline):
            manager = parse_pipeline(pipeline)
            pipelines.append(str(manager))
            return manager

    monkeypatch.setattr(ttl_api, "PassManager", RecordingPassManager)
    return pipelines


@pytest.mark.parametrize(("dtype", "rtol", "atol"), TREE_REDUCE_CASES)
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "specialize_cores", [False, True], ids=["default", "specialized"]
)
def test_tree_reduce(
    device, dtype, rtol, atol, to_device, specialize_cores, monkeypatch
):
    pipelines = _record_pipelines(monkeypatch)
    operation = _make_tree_reduce(dtype == torch.float32, specialize_cores)
    torch.manual_seed(0)
    source_host = torch.randn(TILE_SIZE, CORE_COUNT * TILE_SIZE, dtype=dtype)
    output_host = torch.zeros(OUTPUT_ROWS, TILE_SIZE, dtype=dtype)
    source = to_device(source_host, device)
    output = to_device(output_host, device, tile=(16, TILE_SIZE))

    operation(source, output)
    ttnn.synchronize_device(device)

    # Inspect the actual Python pipeline: C++-only tests cannot detect a
    # missing pass or premature argument finalization in this construction.
    pipeline = next(item for item in pipelines if "convert-ttl-to-ttkernel" in item)
    ordered_passes = (
        "ttkernel-batch-static-pipenet-receives",
        "ttkernel-unroll-static-pipenet-record-loops",
        "lower-affine",
        "ttkernel-cleanup",
        "ttkernel-finalize-tensor-runtime-args",
        "ttl-lower-signpost-to-emitc",
        "convert-ttkernel-to-emitc",
    )
    positions = [pipeline.index(name) for name in ordered_passes]
    assert positions == sorted(positions)
    assert all(pipeline.count(name) == 1 for name in ordered_passes)
    assert ("ttkernel-specialize-cores" in pipeline) == specialize_cores

    actual = ttnn.to_torch(output).reshape(OUTPUT_ROWS, TILE_SIZE).float()
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
        .sum(dim=0)[:OUTPUT_ROWS]
    )
    assert_pcc(expected, actual, threshold=0.999)
    assert_allclose(actual, expected, rtol=rtol, atol=atol)

    for tensor in (source, output):
        ttnn.deallocate(tensor)
