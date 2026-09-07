# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for PipeNet execution with compiler-managed DFB storage."""

import importlib.util
import os
import re

import pytest
import torch

import ttl
from ttlang_test_utils import to_dram, to_l1
from utils.correctness import assert_allclose, assert_pcc

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)
pytestmark = pytest.mark.requires_device

TILE = 32
GRID_SIZE = 2
K_TILES = 2
COMPILER_L1_EXTERNAL_HEADER = os.path.join(
    os.path.dirname(__file__), "..", "include", "compiler_l1_external.hpp"
)


@ttl.operation(grid=(GRID_SIZE, GRID_SIZE))
def compiler_l1_pipe_matmul(lhs, rhs, output):
    lhs_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(0, output_row),
                dst=(slice(0, GRID_SIZE), output_row),
            )
            for output_row in range(GRID_SIZE)
        ]
    )
    rhs_net = ttl.PipeNet(
        [
            ttl.Pipe(
                src=(output_column, 0),
                dst=(output_column, slice(0, GRID_SIZE)),
            )
            for output_column in range(GRID_SIZE)
        ]
    )
    lhs_dfb = ttl.make_dataflow_buffer_like(lhs, shape=(1, 1), block_count=2)
    rhs_dfb = ttl.make_dataflow_buffer_like(rhs, shape=(1, 1), block_count=2)
    output_dfb = ttl.make_dataflow_buffer_like(output, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        with output_dfb.reserve() as output_block:
            output_block.store(
                ttl.block.fill(0, shape=output_block.shape, dtype=output_block.dtype)
            )
            for _reduction_tile in range(K_TILES):
                with lhs_dfb.wait() as lhs_block, rhs_dfb.wait() as rhs_block:
                    output_block += lhs_block @ rhs_block

    @ttl.datamovement()
    def load_lhs():
        _output_column, output_row = ttl.node(dims=2)
        for reduction_tile in range(K_TILES):
            with lhs_dfb.reserve() as lhs_block:

                def receive_then_send(receive_pipe):
                    receive_request = ttl.copy(receive_pipe, lhs_block)

                    def send(send_pipe):
                        ttl.copy(lhs[output_row, reduction_tile], lhs_block).wait()
                        ttl.copy(lhs_block, send_pipe).wait()

                    lhs_net.if_src(send)
                    receive_request.wait()

                lhs_net.if_dst(receive_then_send)

    @ttl.datamovement()
    def load_rhs_and_store_output():
        output_column, _output_row = ttl.node(dims=2)
        for reduction_tile in range(K_TILES):
            with rhs_dfb.reserve() as rhs_block:

                def receive_then_send(receive_pipe):
                    receive_request = ttl.copy(receive_pipe, rhs_block)

                    def send(send_pipe):
                        ttl.copy(rhs[reduction_tile, output_column], rhs_block).wait()
                        ttl.copy(rhs_block, send_pipe).wait()

                    rhs_net.if_src(send)
                    receive_request.wait()

                rhs_net.if_dst(receive_then_send)
        with output_dfb.wait() as output_block:
            ttl.copy(output_block, output[_output_row, output_column]).wait()


# Two-axis distribution exercises PipeNet receive, DFB synchronization, and
# matrix multiplication under both allocator strategies.
@pytest.mark.parametrize(
    "allocation_strategy",
    ["first-fit-decreasing", "best-fit-decreasing"],
    ids=["first-fit", "best-fit"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_compiler_l1_pipe_matmul(
    device,
    dtype,
    to_device,
    allocation_strategy,
    reject_metal_dfb_descriptor_creation,
):
    reject_metal_dfb_descriptor_creation()
    lhs_host = torch.randn(GRID_SIZE * TILE, K_TILES * TILE, dtype=dtype)
    rhs_host = torch.randn(K_TILES * TILE, GRID_SIZE * TILE, dtype=dtype)
    output_tensor = to_device(
        torch.zeros(GRID_SIZE * TILE, GRID_SIZE * TILE, dtype=dtype), device
    )

    compiler_l1_pipe_matmul(
        to_device(lhs_host, device),
        to_device(rhs_host, device),
        output_tensor,
        options=(
            "--ttl-memory-model=compiler-l1 "
            f"--ttl-l1-allocation-strategy={allocation_strategy}"
        ),
    )

    threshold = 0.999 if dtype == torch.bfloat16 else 0.99999
    assert_pcc(
        lhs_host.float() @ rhs_host.float(),
        ttnn.to_torch(output_tensor).float(),
        threshold=threshold,
    )


def _make_external_pipe_copy(data_format):
    @ttl.operation(grid=(2, 1))
    def external_pipe_copy(input_tensor, output_tensor):
        send_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        receive_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        pipe_net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def transfer():
            def send(pipe):
                with send_dfb.reserve() as send_block:
                    ttl.copy(input_tensor[0, 0], send_block).wait()
                with send_dfb.wait() as send_block:
                    ttl.copy(send_block, pipe).wait()

            pipe_net.if_src(send)

            def receive(pipe):
                with receive_dfb.reserve() as receive_block:
                    ttl.copy(pipe, receive_block).wait()
                ttl.call_extern_func(
                    COMPILER_L1_EXTERNAL_HEADER,
                    "compiler_l1_copy_dfb",
                    template_args=[
                        ttl.dfb_descriptor(receive_dfb),
                        ttl.dfb_descriptor(result_dfb),
                    ],
                    dfb_effects=[
                        ttl.DFBEffect.reserve(result_dfb, tiles=1),
                        ttl.DFBEffect.wait(receive_dfb, tiles=1),
                        ttl.DFBEffect.push(result_dfb, tiles=1),
                        ttl.DFBEffect.pop(receive_dfb, tiles=1),
                    ],
                )

            pipe_net.if_dst(receive)

        @ttl.datamovement()
        def store_result():
            node_x, _node_y = ttl.node(dims=2)
            if node_x == 1:
                with result_dfb.wait() as result_block:
                    ttl.copy(result_block, output_tensor[0, 0]).wait()

    return external_pipe_copy


# PipeNet receiver addresses must also bind correctly in typed external DFB calls.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_compiler_l1_pipe_receiver_external_copy(
    device, dtype, to_device, reject_metal_dfb_descriptor_creation
):
    reject_metal_dfb_descriptor_creation()
    input_host = torch.randn(TILE, TILE, dtype=dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros_like(input_host), device)

    _make_external_pipe_copy("bf16" if dtype == torch.bfloat16 else "float32")(
        input_tensor,
        output_tensor,
        options="--ttl-memory-model=compiler-l1",
    )

    assert_allclose(
        ttnn.to_torch(output_tensor).float(), input_host.float(), rtol=0, atol=0
    )


def _make_high_index_pipe(tmp_path, preceding_dfb_count):
    source_lines = [
        "import ttl",
        "@ttl.operation(grid=(2, 1))",
        "def high_index_pipe(input_tensor, output_tensor):",
    ]
    for dfb_index in range(preceding_dfb_count):
        source_lines.append(
            f"    preceding_{dfb_index} = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)"
        )
    source_lines += [
        "    send_dfb = ttl.make_dataflow_buffer_like(input_tensor, shape=(1, 1), block_count=1)",
        "    receive_dfb = ttl.make_dataflow_buffer_like(output_tensor, shape=(1, 1), block_count=1)",
        "    pipe_net = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])",
        "    @ttl.compute()",
        "    def compute():",
        "        pass",
        "    @ttl.datamovement()",
        "    def transfer():",
        "        node_x, _node_y = ttl.node(dims=2)",
        "        if node_x == 0:",
    ]
    for dfb_index in range(preceding_dfb_count):
        source_lines += [
            f"            with preceding_{dfb_index}.reserve() as preceding_block:",
            "                ttl.copy(input_tensor[0, 0], preceding_block).wait()",
            f"            with preceding_{dfb_index}.wait() as _discarded_block:",
            "                pass",
        ]
    source_lines += [
        "        def send(pipe):",
        "            with send_dfb.reserve() as send_block:",
        "                ttl.copy(input_tensor[0, 0], send_block).wait()",
        "            with send_dfb.wait() as send_block:",
        "                ttl.copy(send_block, pipe).wait()",
        "        pipe_net.if_src(send)",
        "        def receive(pipe):",
        "            with receive_dfb.reserve() as receive_block:",
        "                ttl.copy(pipe, receive_block).wait()",
        "            with receive_dfb.wait() as receive_block:",
        "                ttl.copy(receive_block, output_tensor[0, 0]).wait()",
        "        pipe_net.if_dst(receive)",
        "    @ttl.datamovement()",
        "    def unused_transfer():",
        "        pass",
    ]
    source_file = tmp_path / "compiler_l1_high_index_pipe.py"
    source_file.write_text("\n".join(source_lines) + "\n")
    module_spec = importlib.util.spec_from_file_location(
        "compiler_l1_high_index_pipe", source_file
    )
    source_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(source_module)
    return source_module.high_index_pipe


# A receiver above the Metal descriptor limit must execute without constructing
# any TT-Metal DFB descriptor.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_compiler_l1_pipe_receiver_above_metal_index_limit(
    device, dtype, monkeypatch, tmp_path, reject_metal_dfb_descriptor_creation
):
    reject_metal_dfb_descriptor_creation()
    preceding_dfb_count = 65
    operation = _make_high_index_pipe(tmp_path, preceding_dfb_count)
    final_mlir_path = tmp_path / "compiler_l1_high_index_pipe.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    input_host = torch.randn(TILE, TILE, dtype=dtype)
    output_tensor = to_dram(torch.zeros_like(input_host), device)

    for _invocation_index in range(2):
        operation(
            to_dram(input_host, device),
            output_tensor,
            options="--ttl-memory-model=compiler-l1",
        )
        assert_pcc(input_host.float(), ttnn.to_torch(output_tensor).float())

    final_mlir = final_mlir_path.read_text()
    computed_indices_match = re.search(
        r"ttl\.pipe_computed_address_dfb_indices = array<i32: ([^>]*)>",
        final_mlir,
    )
    assert computed_indices_match is not None
    computed_indices = {
        int(index) for index in re.findall(r"\d+", computed_indices_match.group(1))
    }
    assert computed_indices
    assert min(computed_indices) >= preceding_dfb_count
    for computed_index in computed_indices:
        assert f"dfb_index = {computed_index} : i32" in final_mlir
