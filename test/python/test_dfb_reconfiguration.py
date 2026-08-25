# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for synchronized multi-epoch DFB configuration."""

import os

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

SCALAR_RESULT_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "scalar_result_op.hpp"
)
DFB_RECONFIGURATION_TEST_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "dfb_reconfiguration_test_helpers.hpp"
)


def _make_reconfiguration_operation(data_format, grid_cols):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    first_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )
    second_boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(grid_cols, 1))
    def reconfiguration_operation(
        first_input,
        first_output,
        second_input,
        second_output,
        third_input,
        third_output,
    ):
        first_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        first_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        second_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        second_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        third_source = ttl.make_dfb(data_format, shape=(2, 1), block_count=4)
        third_result = ttl.make_dfb(data_format, shape=(2, 1), block_count=4)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with first_source.wait() as source:
                with first_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(first_boundary)
            with second_source.wait() as source:
                with second_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(second_boundary)
            with third_source.wait() as source:
                with third_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, _ = ttl.node(dims=2)
            with first_source.reserve() as destination:
                ttl.copy(first_input[0, node_x], destination).wait()
            ttl.reconfigure_dfbs(first_boundary)
            with second_source.reserve() as destination:
                ttl.copy(
                    second_input[0:1, node_x * 2 : node_x * 2 + 2],
                    destination,
                ).wait()
            ttl.reconfigure_dfbs(second_boundary)
            with third_source.reserve() as destination:
                ttl.copy(third_input[0:2, node_x : node_x + 1], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, _ = ttl.node(dims=2)
            with first_result.wait() as source:
                ttl.copy(source, first_output[0, node_x]).wait()
            ttl.reconfigure_dfbs(first_boundary)
            with second_result.wait() as source:
                ttl.copy(
                    source,
                    second_output[0:1, node_x * 2 : node_x * 2 + 2],
                ).wait()
            ttl.reconfigure_dfbs(second_boundary)
            with third_result.wait() as source:
                ttl.copy(source, third_output[0:2, node_x : node_x + 1]).wait()

    return reconfiguration_operation


def _make_live_crossing_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def live_crossing_operation(
        before_input,
        before_output,
        crossing_input,
        crossing_output,
        after_input,
        after_output,
    ):
        before_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        before_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        crossing_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        crossing_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        after_source = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)
        after_result = ttl.make_dfb(data_format, shape=(1, 2), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def live_payload_compute():
            with before_source.wait() as source:
                with before_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(boundary)
            with crossing_source.wait() as source:
                with crossing_result.reserve() as result:
                    result.store(source)
            with after_source.wait() as source:
                with after_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def live_payload_reader():
            with before_source.reserve() as destination:
                ttl.copy(before_input[0, 0], destination).wait()
            with crossing_source.reserve() as destination:
                ttl.copy(crossing_input[0, 0], destination).wait()
            ttl.reconfigure_dfbs(boundary)
            with after_source.reserve() as destination:
                ttl.copy(after_input[0:1, 0:2], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def live_payload_writer():
            with before_result.wait() as source:
                ttl.copy(source, before_output[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with crossing_result.wait() as source:
                ttl.copy(source, crossing_output[0, 0]).wait()
            with after_result.wait() as source:
                ttl.copy(source, after_output[0:1, 0:2]).wait()

    return live_crossing_operation


def _make_conditional_reconfiguration_operation(data_format, enabled_column):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(2, 1))
    def conditional_reconfiguration_operation(input_tensor, output_tensor):
        source_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            node_x, node_y = ttl.node(dims=2)
            if node_x == enabled_column:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.wait() as source:
                    with result_dfb.reserve() as result:
                        result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            node_x, node_y = ttl.node(dims=2)
            if node_x == enabled_column:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.reserve() as destination:
                    ttl.copy(input_tensor[0, node_x], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            node_x, node_y = ttl.node(dims=2)
            if node_x == enabled_column:
                ttl.reconfigure_dfbs(boundary)
                with result_dfb.wait() as source:
                    ttl.copy(source, output_tensor[0, node_x]).wait()

    return conditional_reconfiguration_operation


def _make_dispatch_condition_reconfiguration_operation(data_format, active_value):
    active = ttl.DispatchCondition(ttl.ScalarType.I32)
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def dispatch_condition_reconfiguration_operation(input_tensor, output_tensor):
        source_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            is_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_constant",
                template_args=[active_value],
                condition_result=active,
            )
            if is_active:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.wait() as source:
                    with result_dfb.reserve() as result:
                        result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            is_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_constant",
                template_args=[active_value],
                condition_result=active,
            )
            if is_active:
                ttl.reconfigure_dfbs(boundary)
                with source_dfb.reserve() as destination:
                    ttl.copy(input_tensor[0, 0], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            is_active = ttl.call_extern_func(
                SCALAR_RESULT_HEADER,
                "scalar_result_constant",
                template_args=[active_value],
                condition_result=active,
            )
            if is_active:
                ttl.reconfigure_dfbs(boundary)
                with result_dfb.wait() as source:
                    ttl.copy(source, output_tensor[0, 0]).wait()

    return dispatch_condition_reconfiguration_operation


def _make_high_index_reconfiguration_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def high_index_reconfiguration_operation(input_tensor, output_tensor):
        padding_dfb_00 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_01 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_02 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_03 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_04 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_05 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_06 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_07 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_08 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_09 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_10 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_11 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_12 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_13 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_14 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_15 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_16 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_17 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_18 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_19 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_20 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_21 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_22 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_23 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_24 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_25 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_26 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_27 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_28 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_29 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_30 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        padding_dfb_31 = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        source_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                DFB_RECONFIGURATION_TEST_HEADER,
                "retain_dfb_liveness",
                template_args=[
                    ttl.dfb_descriptor(padding_dfb_00),
                    ttl.dfb_descriptor(padding_dfb_01),
                    ttl.dfb_descriptor(padding_dfb_02),
                    ttl.dfb_descriptor(padding_dfb_03),
                    ttl.dfb_descriptor(padding_dfb_04),
                    ttl.dfb_descriptor(padding_dfb_05),
                    ttl.dfb_descriptor(padding_dfb_06),
                    ttl.dfb_descriptor(padding_dfb_07),
                    ttl.dfb_descriptor(padding_dfb_08),
                    ttl.dfb_descriptor(padding_dfb_09),
                    ttl.dfb_descriptor(padding_dfb_10),
                    ttl.dfb_descriptor(padding_dfb_11),
                    ttl.dfb_descriptor(padding_dfb_12),
                    ttl.dfb_descriptor(padding_dfb_13),
                    ttl.dfb_descriptor(padding_dfb_14),
                    ttl.dfb_descriptor(padding_dfb_15),
                    ttl.dfb_descriptor(padding_dfb_16),
                    ttl.dfb_descriptor(padding_dfb_17),
                    ttl.dfb_descriptor(padding_dfb_18),
                    ttl.dfb_descriptor(padding_dfb_19),
                    ttl.dfb_descriptor(padding_dfb_20),
                    ttl.dfb_descriptor(padding_dfb_21),
                    ttl.dfb_descriptor(padding_dfb_22),
                    ttl.dfb_descriptor(padding_dfb_23),
                    ttl.dfb_descriptor(padding_dfb_24),
                    ttl.dfb_descriptor(padding_dfb_25),
                    ttl.dfb_descriptor(padding_dfb_26),
                    ttl.dfb_descriptor(padding_dfb_27),
                    ttl.dfb_descriptor(padding_dfb_28),
                    ttl.dfb_descriptor(padding_dfb_29),
                    ttl.dfb_descriptor(padding_dfb_30),
                    ttl.dfb_descriptor(padding_dfb_31),
                ],
                unknown_dfb_access=True,
            )
            ttl.reconfigure_dfbs(boundary)
            with source_dfb.wait() as source:
                with result_dfb.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            ttl.reconfigure_dfbs(boundary)
            with source_dfb.reserve() as destination:
                ttl.copy(input_tensor[0, 0], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reconfigure_dfbs(boundary)
            with result_dfb.wait() as source:
                ttl.copy(source, output_tensor[0, 0]).wait()

    return high_index_reconfiguration_operation


def _make_tensor_backed_reconfiguration_operation(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    boundary = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel)
    )

    @ttl.operation(grid=(1, 1))
    def tensor_backed_reconfiguration_operation(
        tensor_backed_input,
        tensor_backed_output,
        scratch_input,
        scratch_output,
    ):
        tensor_backed_source = ttl.make_tensor_backed_dfb(
            tensor_backed_input, shape=(1, 1)
        )
        tensor_backed_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        scratch_source = ttl.make_dfb(data_format, shape=(1, 1), block_count=3)
        scratch_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=3)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            with tensor_backed_source.wait() as source:
                with tensor_backed_result.reserve() as result:
                    result.store(source)
            ttl.reconfigure_dfbs(boundary)
            with scratch_source.wait() as source:
                with scratch_result.reserve() as result:
                    result.store(source)

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            tensor_backed_source.publish()
            ttl.reconfigure_dfbs(boundary)
            with scratch_source.reserve() as destination:
                ttl.copy(scratch_input[0, 0], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with tensor_backed_result.wait() as source:
                ttl.copy(source, tensor_backed_output[0, 0]).wait()
            ttl.reconfigure_dfbs(boundary)
            with scratch_result.wait() as source:
                ttl.copy(source, scratch_output[0, 0]).wait()

    return tensor_backed_reconfiguration_operation


def _assert_output(actual, expected, dtype):
    tolerance = (0.05, 1.0) if dtype == torch.bfloat16 else (1e-5, 1e-6)
    assert_allclose(
        ttnn.to_torch(actual).float(),
        expected.float(),
        rtol=tolerance[0],
        atol=tolerance[1],
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("grid_cols", [1, 2], ids=["one-core", "two-core"])
@pytest.mark.parametrize(
    "to_device",
    [to_dram, to_l1],
    ids=["dram", "l1"],
)
def test_reconfiguration_reuses_ids_with_different_capacity_and_cached_execution(
    device, dtype, grid_cols, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_reconfiguration_operation(data_format, grid_cols)
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(tmp_path / "reconfiguration.mlir"))

    first_host = (
        torch.arange(32 * 32 * grid_cols, dtype=torch.float32)
        .reshape(32, 32 * grid_cols)
        .to(dtype)
    )
    second_host = (
        torch.arange(32 * 64 * grid_cols, dtype=torch.float32)
        .reshape(32, 64 * grid_cols)
        .remainder(257)
    ).to(dtype)
    third_host = (
        torch.arange(64 * 32 * grid_cols, dtype=torch.float32)
        .reshape(64, 32 * grid_cols)
        .remainder(193)
    ).to(dtype)
    first_output = to_device(torch.zeros_like(first_host), device)
    second_output = to_device(torch.zeros_like(second_host), device)
    third_output = to_device(torch.zeros_like(third_host), device)
    operation(
        to_device(first_host, device),
        first_output,
        to_device(second_host, device),
        second_output,
        to_device(third_host, device),
        third_output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = (tmp_path / "reconfiguration.mlir").read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 2
    assert final_mlir.count("entry_reconfiguration = 0 : i64") == 2
    assert final_mlir.count("entry_reconfiguration = 1 : i64") == 2
    assert final_mlir.count("block_count = 4 : i32") == 2
    assert final_mlir.count("num_tiles = 2 : i32") == 4
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 6

    cached_first_host = (first_host.float() + 3).to(dtype)
    cached_second_host = (second_host.float() - 5).to(dtype)
    cached_third_host = (third_host.float() + 7).to(dtype)
    cached_first_output = to_device(torch.zeros_like(first_host), device)
    cached_second_output = to_device(torch.zeros_like(second_host), device)
    cached_third_output = to_device(torch.zeros_like(third_host), device)
    operation(
        to_device(cached_first_host, device),
        cached_first_output,
        to_device(cached_second_host, device),
        cached_second_output,
        to_device(cached_third_host, device),
        cached_third_output,
        options="--ttl-reuse-user-dfbs",
    )

    for actual, expected in (
        (first_output, first_host),
        (second_output, second_host),
        (third_output, third_host),
        (cached_first_output, cached_first_host),
        (cached_second_output, cached_second_host),
        (cached_third_output, cached_third_host),
    ):
        _assert_output(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_live_payload_crosses_reconfiguration_and_cached_execution(
    device, dtype, to_device, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_live_crossing_operation(data_format)
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(tmp_path / "live_crossing.mlir"))

    initial_inputs = (
        torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32).to(dtype),
        torch.arange(32 * 32, dtype=torch.float32)
        .reshape(32, 32)
        .remainder(251)
        .to(dtype),
        torch.arange(32 * 64, dtype=torch.float32)
        .reshape(32, 64)
        .remainder(197)
        .to(dtype),
    )
    initial_outputs = tuple(
        to_device(torch.zeros_like(input_tensor), device)
        for input_tensor in initial_inputs
    )
    operation(
        to_device(initial_inputs[0], device),
        initial_outputs[0],
        to_device(initial_inputs[1], device),
        initial_outputs[1],
        to_device(initial_inputs[2], device),
        initial_outputs[2],
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = (tmp_path / "live_crossing.mlir").read_text()
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
    assert "ttl.dfb_reconfiguration_plan" in final_mlir
    for actual, expected in zip(initial_outputs, initial_inputs):
        _assert_output(actual, expected, dtype)

    cached_inputs = tuple(
        (input_tensor.float() + offset).to(dtype)
        for input_tensor, offset in zip(initial_inputs, (3, -5, 7))
    )
    cached_outputs = tuple(
        to_device(torch.zeros_like(input_tensor), device)
        for input_tensor in cached_inputs
    )
    operation(
        to_device(cached_inputs[0], device),
        cached_outputs[0],
        to_device(cached_inputs[1], device),
        cached_outputs[1],
        to_device(cached_inputs[2], device),
        cached_outputs[2],
        options="--ttl-reuse-user-dfbs",
    )

    for actual, expected in zip(cached_outputs, cached_inputs):
        _assert_output(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("enabled_column", [0, 1], ids=["left", "right"])
def test_conditional_reconfiguration_executes_with_post_boundary_dfbs(
    device,
    dtype,
    to_device,
    enabled_column,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_conditional_reconfiguration_operation(data_format, enabled_column)
    monkeypatch.setenv(
        "TTLANG_FINAL_MLIR",
        str(tmp_path / f"conditional_{enabled_column}.mlir"),
    )
    input_host = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64).to(dtype)
    output = to_device(torch.zeros_like(input_host), device)
    operation(
        to_device(input_host, device),
        output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = (tmp_path / f"conditional_{enabled_column}.mlir").read_text()
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
    expected = torch.zeros_like(input_host)
    column_start = enabled_column * 32
    expected[:, column_start : column_start + 32] = input_host[
        :, column_start : column_start + 32
    ]
    _assert_output(output, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_dispatch_condition_reconfiguration_executes_active_and_inactive(
    device,
    dtype,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    for active_value in (1, 0):
        operation = _make_dispatch_condition_reconfiguration_operation(
            data_format, active_value
        )
        mlir_file = tmp_path / f"dispatch_condition_{active_value}.mlir"
        monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))
        for invocation in range(2):
            input_host = (
                torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
                + invocation * 7
            ).to(dtype)
            output = to_device(torch.zeros_like(input_host), device)
            operation(
                to_device(input_host, device),
                output,
                options="--ttl-reuse-user-dfbs",
            )
            expected = input_host if active_value else torch.zeros_like(input_host)
            _assert_output(output, expected, dtype)

        final_mlir = mlir_file.read_text()
        assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
        assert final_mlir.count("scalar_result_constant") == 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_reconfiguration_executes_with_physical_indices_above_31(
    device,
    dtype,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires the Blackhole 64-index DFB capacity")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_high_index_reconfiguration_operation(data_format)
    mlir_file = tmp_path / "high_index_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))
    input_host = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32).to(dtype)
    output = to_device(torch.zeros_like(input_host), device)
    operation(
        to_device(input_host, device),
        output,
        options="--ttl-reuse-user-dfbs",
    )

    final_mlir = mlir_file.read_text()
    assert "dfb_index = 32 : i32" in final_mlir
    assert "dfb_index = 33 : i32" in final_mlir
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
    _assert_output(output, input_host, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
def test_reconfiguration_switches_tensor_backed_storage_and_cached_execution(
    device,
    dtype,
    to_device,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    data_format = "bf16" if dtype == torch.bfloat16 else "float32"
    operation = _make_tensor_backed_reconfiguration_operation(data_format)
    mlir_file = tmp_path / "tensor_backed_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(mlir_file))

    for invocation in range(2):
        tensor_backed_host = (
            torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32) + invocation * 3
        ).to(dtype)
        scratch_host = (
            torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
            + invocation * 5
            + 11
        ).to(dtype)
        tensor_backed_output = to_device(torch.zeros_like(tensor_backed_host), device)
        scratch_output = to_device(torch.zeros_like(scratch_host), device)
        operation(
            to_l1_sharded(tensor_backed_host, device, layout="height"),
            tensor_backed_output,
            to_device(scratch_host, device),
            scratch_output,
            options="--ttl-reuse-user-dfbs",
        )
        _assert_output(tensor_backed_output, tensor_backed_host, dtype)
        _assert_output(scratch_output, scratch_host, dtype)

    final_mlir = mlir_file.read_text()
    allocation_metadata = final_mlir.partition("ttl.dfb_reconfiguration_plan")[0]
    assert allocation_metadata.count("dfb_index = ") == 2
    assert "tensor_index = 0" in final_mlir
    assert final_mlir.count("entry_reconfiguration = 0 : i64") == 2
    assert final_mlir.count("experimental::reconfigure_dfb_interfaces") == 3
