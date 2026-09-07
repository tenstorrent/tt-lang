# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for compiler-managed L1 lifecycle boundaries and externs."""

import importlib.util
import os
import re

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32
SCALAR_RESULT_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "scalar_result_op.hpp"
)
COMPILER_L1_EXTERNAL_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "compiler_l1_external.hpp"
)


def _data_format(dtype):
    return "bf16" if dtype == torch.bfloat16 else "float32"


def _assert_exact(actual, expected):
    assert_allclose(actual.float(), expected.float(), rtol=0, atol=0)


def _make_scalar_external_compute(data_format):
    @ttl.operation(grid=(1, 1))
    def scalar_external_compute(input_tensor, output_tensor):
        input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute()
        def compute():
            active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_predicate",
                template_args=[True],
                result_type=ttl.ScalarType.I32,
            )
            if active:
                with input_dfb.wait() as input_block:
                    with output_dfb.reserve() as output_block:
                        output_block.store(input_block)

        @ttl.datamovement()
        def read():
            with input_dfb.reserve() as input_block:
                ttl.copy(input_tensor[0, 0], input_block).wait()

        @ttl.datamovement()
        def write():
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return scalar_external_compute


def _make_external_selected_reset(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(1, 1))
    def external_selected_reset(input_tensor, output_tensor):
        stale_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        shared_allocation = ttl.make_dfb_allocation_group()
        reset_target = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=2,
            allocation_group=shared_allocation,
        )
        current_source = ttl.make_dfb(
            data_format,
            shape=(1, 1),
            block_count=2,
            allocation_group=shared_allocation,
        )
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reset_dfbs(reset, dfbs=[reset_target])
            with current_source.wait() as current_block:
                with output_dfb.reserve() as output_block:
                    output_block.store(current_block)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with stale_source.reserve() as stale_block:
                ttl.copy(input_tensor[0, 0], stale_block).wait()
            ttl.reset_dfbs(reset, dfbs=[reset_target])
            with current_source.reserve() as current_block:
                ttl.copy(input_tensor[0, 1], current_block).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.call_extern_func(
                COMPILER_L1_EXTERNAL_HEADER,
                "compiler_l1_copy_dfb",
                template_args=[
                    ttl.dfb_descriptor(stale_source),
                    ttl.dfb_descriptor(reset_target),
                ],
                dfb_effects=[
                    ttl.DFBEffect.reserve(reset_target, tiles=1),
                    ttl.DFBEffect.wait(stale_source, tiles=1),
                    ttl.DFBEffect.push(reset_target, tiles=1),
                    ttl.DFBEffect.pop(stale_source, tiles=1),
                ],
            )
            ttl.reset_dfbs(reset, dfbs=[reset_target])
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return external_selected_reset


def _make_reset_all(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(1, 1))
    def reset_all(input_tensor, output_tensor):
        stale_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        current_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=3)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.reset_all_dfbs(reset)
            with current_dfb.wait() as current_block:
                with output_dfb.reserve() as output_block:
                    output_block.store(current_block)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with stale_dfb.reserve() as stale_block:
                ttl.copy(input_tensor[0, 0], stale_block).wait()
            ttl.reset_all_dfbs(reset)
            with current_dfb.reserve() as current_block:
                ttl.copy(input_tensor[0, 1], current_block).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reset_all_dfbs(reset)
            with output_dfb.wait() as output_block:
                ttl.copy(output_block, output_tensor[0, 0]).wait()

    return reset_all


def _make_reconfiguration(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def reconfiguration(
        before_input,
        before_output,
        preserved_input,
        preserved_output,
        after_input,
        after_output,
    ):
        before_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        before_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        preserved_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        preserved_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        after_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        after_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with before_source.wait() as source:
                with before_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(boundary)
            with preserved_source.wait() as source:
                with preserved_result.reserve() as result:
                    result.store(source)
            with after_source.wait() as source:
                with after_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with before_source.reserve() as destination:
                ttl.copy(before_input[0, 0], destination).wait()
            with preserved_source.reserve() as destination:
                ttl.copy(preserved_input[0, 0], destination).wait()
            ttl.reconfigure_dfbs(boundary)
            with after_source.reserve() as destination:
                ttl.copy(after_input[0:1, 0:2], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with before_result.wait() as source:
                ttl.copy(source, before_output[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with preserved_result.wait() as source:
                ttl.copy(source, preserved_output[0, 0]).wait()
            with after_result.wait() as source:
                ttl.copy(source, after_output[0:1, 0:2]).wait()

    return reconfiguration


def _make_high_index_reset(tmp_path, data_format, dfb_count):
    lines = [
        "import ttl",
        "compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)",
        "reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)",
        "writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)",
        "reset = ttl.DFBReset(participants=(compute_kernel, reader_kernel, writer_kernel))",
        "@ttl.operation(grid=(1, 1))",
        "def high_index_reset(input_tensor, output_tensor):",
    ]
    for dfb_index in range(dfb_count):
        lines.append(
            f'    storage_{dfb_index} = ttl.make_dfb("{data_format}", shape=(1, 1), block_count=1)'
        )
    reset_target = f"storage_{dfb_count - 1}"
    lines += [
        f'    output_dfb = ttl.make_dfb("{data_format}", shape=(1, 1), block_count=1)',
        "    @ttl.compute(kernel=compute_kernel)",
        "    def compute():",
    ]
    for dfb_index in range(dfb_count - 1):
        lines += [
            f"        with storage_{dfb_index}.wait() as source:",
            "            pass",
        ]
    lines += [
        f"        ttl.reset_dfbs(reset, dfbs=[{reset_target}])",
        f"        with {reset_target}.wait() as source:",
        "            with output_dfb.reserve() as destination:",
        "                destination.store(source)",
        "    @ttl.datamovement(kernel=reader_kernel)",
        "    def read():",
    ]
    for dfb_index in range(dfb_count - 1):
        lines += [
            f"        with storage_{dfb_index}.reserve() as destination:",
            "            ttl.copy(input_tensor[0, 0], destination).wait()",
        ]
    lines += [
        f"        with {reset_target}.reserve() as destination:",
        "            ttl.copy(input_tensor[0, 0], destination).wait()",
        f"        ttl.reset_dfbs(reset, dfbs=[{reset_target}])",
        f"        with {reset_target}.reserve() as destination:",
        "            ttl.copy(input_tensor[0, 1], destination).wait()",
        "    @ttl.datamovement(kernel=writer_kernel)",
        "    def write():",
        f"        ttl.reset_dfbs(reset, dfbs=[{reset_target}])",
        "        with output_dfb.wait() as source:",
        "            ttl.copy(source, output_tensor[0, 0]).wait()",
    ]
    lines = [
        lines[0],
        "def make():",
        *(f"    {line}" for line in lines[1:]),
        "    return high_index_reset",
    ]
    source_file = tmp_path / "high_index_reset.py"
    source_file.write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location("high_index_reset", source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_compiler_l1_scalar_external_compute(device, dtype, to_device):
    expected = torch.randn(TILE, TILE, dtype=dtype)
    input_tensor = to_device(expected, device)
    output_tensor = to_device(torch.zeros_like(expected), device)

    _make_scalar_external_compute(_data_format(dtype))(
        input_tensor,
        output_tensor,
        options="--ttl-memory-model=compiler-l1",
    )

    _assert_exact(ttnn.to_torch(output_tensor), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_compiler_l1_external_selected_reset(
    device, dtype, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole synchronized DFB reset support")
    input_host = torch.randn(TILE, 2 * TILE, dtype=dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros((TILE, TILE), dtype=dtype), device)
    final_mlir = tmp_path / "compiler_l1_external_group_reset.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))

    operation = _make_external_selected_reset(_data_format(dtype))
    for _invocation_index in range(2):
        operation(
            input_tensor,
            output_tensor,
            options="--ttl-memory-model=compiler-l1",
        )
        _assert_exact(ttnn.to_torch(output_tensor), input_host[:, TILE:])

    allocation_line = final_mlir.read_text().splitlines()[0]
    storage_records = re.findall(
        r"dfb_index = \d+.*?l1_offset = (\d+).*?"
        r"l1_payload_offset = (\d+).*?storage_index = (\d+)",
        allocation_line,
    )
    assert len(storage_records) == 4
    assert len(set(storage_records)) == 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_compiler_l1_reset_all(device, dtype, to_device):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole synchronized DFB reset support")
    input_host = torch.randn(TILE, 2 * TILE, dtype=dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros((TILE, TILE), dtype=dtype), device)

    operation = _make_reset_all(_data_format(dtype))
    for _invocation_index in range(2):
        operation(
            input_tensor,
            output_tensor,
            options="--ttl-memory-model=compiler-l1",
        )
        _assert_exact(ttnn.to_torch(output_tensor), input_host[:, TILE:])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_compiler_l1_reconfiguration(device, dtype, to_device, monkeypatch, tmp_path):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")
    before_host = torch.randn(TILE, TILE, dtype=dtype)
    preserved_host = torch.randn(TILE, TILE, dtype=dtype)
    after_host = torch.randn(TILE, 2 * TILE, dtype=dtype)
    inputs = [
        to_device(before_host, device),
        to_device(preserved_host, device),
        to_device(after_host, device),
    ]
    outputs = [
        to_device(torch.zeros_like(before_host), device),
        to_device(torch.zeros_like(preserved_host), device),
        to_device(torch.zeros_like(after_host), device),
    ]
    final_mlir = tmp_path / "compiler_l1_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))

    operation = _make_reconfiguration(_data_format(dtype))
    for _invocation_index in range(2):
        operation(
            inputs[0],
            outputs[0],
            inputs[1],
            outputs[1],
            inputs[2],
            outputs[2],
            options="--ttl-memory-model=compiler-l1",
        )
        for actual, expected in zip(outputs, [before_host, preserved_host, after_host]):
            _assert_exact(ttnn.to_torch(actual), expected)
    offsets = [
        int(offset)
        for offset in re.findall(r"l1_payload_offset = (\d+)", final_mlir.read_text())
    ]
    assert len(offsets) == 6
    assert len(set(offsets)) < len(offsets)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_compiler_l1_reset_above_metal_index_limit(
    device, dtype, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole synchronized DFB reset support")
    metal_index_capacity = 64
    expected_reset_index = metal_index_capacity + 1
    dfb_count = expected_reset_index + 3
    input_host = torch.randn(TILE, 2 * TILE, dtype=dtype)
    input_tensor = to_device(input_host, device)
    output_tensor = to_device(torch.zeros((TILE, TILE), dtype=dtype), device)
    final_mlir = tmp_path / "compiler_l1_high_index_reset.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))
    operation = _make_high_index_reset(tmp_path, _data_format(dtype), dfb_count)

    for _invocation_index in range(2):
        operation(
            input_tensor,
            output_tensor,
            options="--ttl-memory-model=compiler-l1",
        )
        _assert_exact(ttnn.to_torch(output_tensor), input_host[:, TILE:])

    final_ir = final_mlir.read_text()
    assert final_ir.count("l1_payload_offset =") == dfb_count + 1
    reset_indices = {
        int(index)
        for index in re.findall(
            r"ttkernel\.cb_ctarg_idx = (\d+) : i32\} : ui32\n"
            r'\s+emitc\.call_opaque "ttlang::l1::resetState"',
            final_ir,
        )
    }
    assert reset_indices == {expected_reset_index}
