# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for DFB allocation with external C++ operations.

The external multiply receives typed DFB descriptors, which are direct DFB
dependencies, while the C++ body owns its compute-thread DFB protocol.
The larger inlinable atom composition separates two calls with native data
movement and compute. The allocator cannot inspect the external DFB accesses,
so DFBs whose protocol exists only in C++ remain conservatively unbounded.
"""

import os
import re

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

# TTNN interop rejects non-tilized tensors before DFB lowering, so TILE is the
# only supported tensor layout for these runtime cases.
TILE = 32
OVER_CAPACITY_COMPOSITION_LEVELS = 6
EXTERNAL_COMPOSITION_LOGICAL_DFBS = (1 << OVER_CAPACITY_COMPOSITION_LEVELS) + 6
EXTERNAL_COMPOSITION_PHYSICAL_DFBS = 8
# The external fp32 pack step produces approximately 1/256 output increments.
# These bounds cover measured relative errors of 3.51e-3 for multiplication and
# 1.76e-3 for the composition while retaining small absolute bounds near zero.
F32_EXTERNAL_MULTIPLY_RTOL = 5e-3
F32_EXTERNAL_MULTIPLY_ATOL = 1e-4
F32_EXTERNAL_COMPOSITION_RTOL = 2e-3
F32_EXTERNAL_COMPOSITION_ATOL = 5e-4
EXTERNAL_MULTIPLY_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "external_eltwise_mul.hpp"
)
METAL_STORAGE_OPTIONS = "--ttl-memory-model=metal-cb"
EXTERNAL_STORAGE_OPTIONS = [
    pytest.param(METAL_STORAGE_OPTIONS, id="metal"),
    pytest.param(
        "--ttl-memory-model=compiler-l1 "
        "--ttl-l1-allocation-strategy=first-fit-decreasing",
        id="compiler-l1-first-fit",
    ),
    pytest.param(
        "--ttl-memory-model=compiler-l1 "
        "--ttl-l1-allocation-strategy=best-fit-decreasing",
        id="compiler-l1-best-fit",
    ),
]


@ttl.operation()
def _external_eltwise_mul(lhs: ttl.DFB, rhs: ttl.DFB, result: ttl.DFB):
    call_extern_func(
        EXTERNAL_MULTIPLY_HEADER,
        "ttl_external_eltwise_mul",
        template_args=[
            dfb_descriptor(lhs),
            dfb_descriptor(rhs),
            dfb_descriptor(result),
        ],
        kernel=ttl.KernelKind.COMPUTE,
    )


def _make_external_multiply_kernel(data_format):
    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
    def external_multiply_kernel(lhs, rhs, result):
        lhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        rhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        lhs_destination = lhs_dfb.reserve()
        ttl.copy(lhs[0, 0], lhs_destination).wait()
        lhs_destination.push()
        rhs_destination = rhs_dfb.reserve()
        ttl.copy(rhs[0, 0], rhs_destination).wait()
        rhs_destination.push()

        _external_eltwise_mul(lhs_dfb, rhs_dfb, result_dfb)

        result_source = result_dfb.wait()
        ttl.copy(result_source, result[0, 0]).wait()
        result_source.pop()

    return external_multiply_kernel


def _make_external_reset_kernel(data_format, distinct_logical_dfbs):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
    def external_reset_kernel(lhs, rhs, result):
        lhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        rhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        if distinct_logical_dfbs:
            shared_allocation = ttl.make_dfb_allocation_group()
            stale_result = ttl.make_dfb(
                data_format,
                shape=(1, 1),
                block_count=2,
                allocation_group=shared_allocation,
            )
            current_result = ttl.make_dfb(
                data_format,
                shape=(1, 1),
                block_count=2,
                allocation_group=shared_allocation,
            )
        else:
            stale_result = ttl.make_dfb(
                data_format,
                shape=(1, 1),
                block_count=2,
            )
            current_result = stale_result

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                EXTERNAL_MULTIPLY_HEADER,
                "ttl_external_eltwise_mul",
                template_args=[
                    ttl.dfb_descriptor(lhs_dfb),
                    ttl.dfb_descriptor(rhs_dfb),
                    ttl.dfb_descriptor(stale_result),
                ],
            )
            ttl.reset_dfbs(reset, dfbs=[stale_result])

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with lhs_dfb.reserve() as lhs_destination:
                ttl.copy(lhs[0, 0], lhs_destination).wait()
            with rhs_dfb.reserve() as rhs_destination:
                ttl.copy(rhs[0, 0], rhs_destination).wait()
            ttl.reset_dfbs(reset, dfbs=[stale_result])
            with current_result.reserve() as current_destination:
                ttl.copy(rhs[0, 0], current_destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reset_dfbs(reset, dfbs=[stale_result])
            with current_result.wait() as current_source:
                ttl.copy(current_source, result[0, 0]).wait()

    return external_reset_kernel


def _make_external_reconfiguration_kernel(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reconfiguration = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=True,
    )

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
    def external_reconfiguration_kernel(lhs, rhs, result):
        before_lhs = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        before_rhs = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        before_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        after_lhs = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        after_rhs = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        after_result = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                EXTERNAL_MULTIPLY_HEADER,
                "ttl_external_eltwise_mul",
                template_args=[
                    ttl.dfb_descriptor(before_lhs),
                    ttl.dfb_descriptor(before_rhs),
                    ttl.dfb_descriptor(before_result),
                ],
                dfb_effects=[
                    ttl.DFBEffect.reserve(before_result, tiles=1),
                    ttl.DFBEffect.wait(before_lhs, tiles=1),
                    ttl.DFBEffect.wait(before_rhs, tiles=1),
                    ttl.DFBEffect.pop(before_lhs, tiles=1),
                    ttl.DFBEffect.pop(before_rhs, tiles=1),
                    ttl.DFBEffect.push(before_result, tiles=1),
                ],
            )
            ttl.reconfigure_dfbs(reconfiguration)
            ttl.call_extern_func(
                EXTERNAL_MULTIPLY_HEADER,
                "ttl_external_eltwise_mul",
                template_args=[
                    ttl.dfb_descriptor(after_lhs),
                    ttl.dfb_descriptor(after_rhs),
                    ttl.dfb_descriptor(after_result),
                ],
                dfb_effects=[
                    ttl.DFBEffect.reserve(after_result, tiles=1),
                    ttl.DFBEffect.wait(after_lhs, tiles=1),
                    ttl.DFBEffect.wait(after_rhs, tiles=1),
                    ttl.DFBEffect.pop(after_lhs, tiles=1),
                    ttl.DFBEffect.pop(after_rhs, tiles=1),
                    ttl.DFBEffect.push(after_result, tiles=1),
                ],
            )

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with before_lhs.reserve() as destination:
                ttl.copy(lhs[0, 0], destination).wait()
            with before_rhs.reserve() as destination:
                ttl.copy(rhs[0, 0], destination).wait()
            ttl.reconfigure_dfbs(reconfiguration)
            with after_lhs.reserve() as destination:
                ttl.copy(lhs[0, 1], destination).wait()
            with after_rhs.reserve() as destination:
                ttl.copy(rhs[0, 1], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with before_result.wait() as source:
                ttl.copy(source, result[0, 0]).wait()
            ttl.reconfigure_dfbs(reconfiguration)
            with after_result.wait() as source:
                ttl.copy(source, result[0, 1]).wait()

    return external_reconfiguration_kernel


def _make_nested_copy_atom(data_format, level_count):
    @ttl.operation()
    def copy_stage(source: ttl.DFB, destination: ttl.DFB):
        destination_block = destination.reserve()
        destination_block.store(source.wait())

    nested_copy = copy_stage
    for _composition_level in range(level_count):
        inner_copy = nested_copy

        @ttl.operation()
        def doubled_copy(source: ttl.DFB, destination: ttl.DFB):
            intermediate_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            inner_copy(source, intermediate_dfb)
            inner_copy(intermediate_dfb, destination)

        nested_copy = doubled_copy

    return nested_copy


def _make_external_composition_body(data_format):
    nested_copy = _make_nested_copy_atom(data_format, OVER_CAPACITY_COMPOSITION_LEVELS)

    @ttl.operation()
    def external_composition_body(
        lhs_dfb: ttl.DFB,
        first_rhs_dfb: ttl.DFB,
        second_rhs_dfb: ttl.DFB,
        result_dfb: ttl.DFB,
    ):
        product_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        copied_product_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        computed_product_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        _external_eltwise_mul(lhs_dfb, first_rhs_dfb, product_dfb)
        nested_copy(product_dfb, copied_product_dfb)

        copied_product = copied_product_dfb.wait()
        computed_product = computed_product_dfb.reserve()
        computed_product.store(ttl.exp(copied_product))
        computed_product.push()

        _external_eltwise_mul(computed_product_dfb, second_rhs_dfb, result_dfb)

    return external_composition_body


def _make_external_composition_kernel(data_format, tensor_backed):
    external_composition_body = _make_external_composition_body(data_format)

    if tensor_backed:

        @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
        def external_composition_kernel(lhs, first_rhs, second_rhs, result):
            lhs_dfb = ttl.make_tensor_backed_dfb(lhs, shape=(1, 1))
            first_rhs_dfb = ttl.make_tensor_backed_dfb(first_rhs, shape=(1, 1))
            second_rhs_dfb = ttl.make_tensor_backed_dfb(second_rhs, shape=(1, 1))
            result_dfb = ttl.make_tensor_backed_dfb(result, shape=(1, 1))

            lhs_dfb.publish()
            first_rhs_dfb.publish()
            second_rhs_dfb.publish()

            external_composition_body(
                lhs_dfb, first_rhs_dfb, second_rhs_dfb, result_dfb
            )

            result_source = result_dfb.wait()
            result_source.pop(kernel=ttl.KernelKind.DATA_MOVEMENT)

    else:

        @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
        def external_composition_kernel(lhs, first_rhs, second_rhs, result):
            lhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            first_rhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            second_rhs_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
            result_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

            lhs_destination = lhs_dfb.reserve()
            ttl.copy(lhs[0, 0], lhs_destination).wait()
            lhs_destination.push()
            first_rhs_destination = first_rhs_dfb.reserve()
            ttl.copy(first_rhs[0, 0], first_rhs_destination).wait()
            first_rhs_destination.push()
            second_rhs_destination = second_rhs_dfb.reserve()
            ttl.copy(second_rhs[0, 0], second_rhs_destination).wait()
            second_rhs_destination.push()

            external_composition_body(
                lhs_dfb, first_rhs_dfb, second_rhs_dfb, result_dfb
            )

            result_source = result_dfb.wait()
            ttl.copy(result_source, result[0, 0]).wait()
            result_source.pop()

    return external_composition_kernel


_external_bf16_multiply = _make_external_multiply_kernel("bf16")
_external_f32_multiply = _make_external_multiply_kernel("float32")
_external_bf16_reset_same_dfb = _make_external_reset_kernel("bf16", False)
_external_f32_reset_same_dfb = _make_external_reset_kernel("float32", False)
_external_bf16_composition = _make_external_composition_kernel("bf16", False)
_external_f32_composition = _make_external_composition_kernel("float32", False)
_tensor_backed_bf16_composition = _make_external_composition_kernel("bf16", True)
_tensor_backed_f32_composition = _make_external_composition_kernel("float32", True)
_external_bf16_reconfiguration = _make_external_reconfiguration_kernel("bf16")
_external_f32_reconfiguration = _make_external_reconfiguration_kernel("float32")

assert EXTERNAL_COMPOSITION_LOGICAL_DFBS > 64


def _count_final_dfb_allocations(final_mlir_path):
    final_mlir = final_mlir_path.read_text()
    return final_mlir.count("dfb_index =")


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_external_bf16_reset_same_dfb, torch.bfloat16),
        (_external_f32_reset_same_dfb, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    "to_device",
    [to_dram, to_l1],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize("storage_options", EXTERNAL_STORAGE_OPTIONS)
def test_external_protocol_state_reset_drains_compute_interfaces(
    device, operation, dtype, to_device, storage_options
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)
    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(rhs_host), device)

    for _invocation_index in range(2):
        operation(
            lhs,
            rhs,
            result,
            options=f"--ttl-reuse-user-dfbs --ttl-specialize-cores {storage_options}",
        )

    actual = ttnn.to_torch(result).float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, rhs_host.float(), rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, rhs_host.float(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_external_bf16_multiply, torch.bfloat16),
        (_external_f32_multiply, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("memory_config", "to_device"),
    [("dram", to_dram), ("l1", to_l1)],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize("reuse_user_dfbs", [True, False], ids=["reuse", "distinct"])
@pytest.mark.parametrize(
    "specialize_cores", [False, True], ids=["generic-cores", "specialized-cores"]
)
@pytest.mark.parametrize("storage_options", EXTERNAL_STORAGE_OPTIONS)
def test_external_multiply_with_dfb_allocation(
    device,
    operation,
    dtype,
    memory_config,
    to_device,
    reuse_user_dfbs,
    specialize_cores,
    storage_options,
):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)

    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(lhs_host), device)

    reuse_option = (
        "--ttl-reuse-user-dfbs" if reuse_user_dfbs else "--no-ttl-reuse-user-dfbs"
    )
    specialization_option = (
        "--ttl-specialize-cores" if specialize_cores else "--no-ttl-specialize-cores"
    )
    operation(
        lhs,
        rhs,
        result,
        options=f"{reuse_option} {specialization_option} {storage_options}",
    )

    actual = ttnn.to_torch(result).float()
    expected = lhs_host.float() * rhs_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(
            actual,
            expected,
            rtol=F32_EXTERNAL_MULTIPLY_RTOL,
            atol=F32_EXTERNAL_MULTIPLY_ATOL,
        )


@pytest.mark.parametrize(
    ("data_format", "dtype"),
    [
        ("bf16", torch.bfloat16),
        ("float32", torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    "to_device",
    [to_dram, to_l1],
    ids=["dram", "l1"],
)
@pytest.mark.parametrize("storage_options", EXTERNAL_STORAGE_OPTIONS)
def test_external_protocol_state_reset_allows_group_reuse(
    device, data_format, dtype, to_device, storage_options, monkeypatch, tmp_path
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    operation = _make_external_reset_kernel(data_format, True)
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)
    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(rhs_host), device)
    final_mlir_path = tmp_path / "external_reset_reuse.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))

    for _invocation_index in range(2):
        operation(
            lhs,
            rhs,
            result,
            options=f"--ttl-reuse-user-dfbs --ttl-specialize-cores {storage_options}",
        )

    final_mlir = final_mlir_path.read_text()
    if storage_options == METAL_STORAGE_OPTIONS:
        assert _count_final_dfb_allocations(final_mlir_path) == 3
    else:
        assert _count_final_dfb_allocations(final_mlir_path) == 4
        storage_indices = re.findall(r"storage_index = (\d+)", final_mlir)
        assert len(storage_indices) == 4
        assert len(set(storage_indices)) == 3
    actual = ttnn.to_torch(result).float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, rhs_host.float(), rtol=0.05, atol=1.0)
    else:
        assert_allclose(actual, rhs_host.float(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("operation", "dtype"),
    [
        (_external_bf16_reconfiguration, torch.bfloat16),
        (_external_f32_reconfiguration, torch.float32),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "allocation_strategy",
    ["first-fit-decreasing", "best-fit-decreasing"],
    ids=["first-fit", "best-fit"],
)
def test_compiler_l1_external_compute_across_reconfiguration(
    device,
    operation,
    dtype,
    to_device,
    allocation_strategy,
    reject_metal_dfb_descriptor_creation,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    reject_metal_dfb_descriptor_creation()
    element_indices = torch.arange(TILE * 2 * TILE, dtype=torch.float32).reshape(
        TILE, 2 * TILE
    )
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)
    lhs = to_device(lhs_host, device)
    rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(lhs_host), device)
    final_mlir_path = tmp_path / "external_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    options = (
        "--ttl-memory-model=compiler-l1 "
        f"--ttl-l1-allocation-strategy={allocation_strategy}"
    )

    for _invocation_index in range(2):
        operation(lhs, rhs, result, options=options)

    final_mlir = final_mlir_path.read_text()
    assert _count_final_dfb_allocations(final_mlir_path) == 6
    payload_offsets = re.findall(r"l1_payload_offset = (\d+)", final_mlir)
    assert len(payload_offsets) == 6
    assert len(set(payload_offsets)) == 3
    actual = ttnn.to_torch(result).float()
    expected = lhs_host.float() * rhs_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(
            actual,
            expected,
            rtol=F32_EXTERNAL_MULTIPLY_RTOL,
            atol=F32_EXTERNAL_MULTIPLY_ATOL,
        )


def _make_external_composition_tensors(device, dtype, to_device):
    element_indices = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE)
    lhs_host = ((element_indices.remainder(41) - 20) / 16).to(dtype)
    rhs_host = (((3 * element_indices).remainder(37) - 18) / 16).to(dtype)
    lhs = to_device(lhs_host, device)
    first_rhs = to_device(rhs_host, device)
    second_rhs = to_device(rhs_host, device)
    result = to_device(torch.zeros_like(lhs_host), device)
    return lhs_host, rhs_host, lhs, first_rhs, second_rhs, result


def _assert_external_composition_result(result, lhs_host, rhs_host, dtype):
    actual = ttnn.to_torch(result).float()
    expected = torch.exp(lhs_host.float() * rhs_host.float()) * rhs_host.float()
    if dtype == torch.bfloat16:
        assert_allclose(actual, expected, rtol=0.05, atol=1.0)
    else:
        assert_allclose(
            actual,
            expected,
            rtol=F32_EXTERNAL_COMPOSITION_RTOL,
            atol=F32_EXTERNAL_COMPOSITION_ATOL,
        )


@pytest.mark.parametrize(
    ("scratch_operation", "tensor_backed_operation", "dtype"),
    [
        (
            _external_bf16_composition,
            _tensor_backed_bf16_composition,
            torch.bfloat16,
        ),
        (
            _external_f32_composition,
            _tensor_backed_f32_composition,
            torch.float32,
        ),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("storage_kind", "to_device"),
    [
        ("scratch", to_dram),
        ("scratch", to_l1),
        ("tensor_backed", to_l1_sharded),
    ],
    ids=["scratch-dram", "scratch-l1", "tensor-backed"],
)
def test_external_composition_requires_dfb_reuse(
    device,
    scratch_operation,
    tensor_backed_operation,
    dtype,
    storage_kind,
    to_device,
    monkeypatch,
    tmp_path,
):
    lhs_host, rhs_host, lhs, first_rhs, second_rhs, result = (
        _make_external_composition_tensors(device, dtype, to_device)
    )
    operation = (
        tensor_backed_operation
        if storage_kind == "tensor_backed"
        else scratch_operation
    )

    # The disabled mode requires one physical index for each logical DFB,
    # exceeding every supported target capacity.
    with pytest.raises(
        RuntimeError,
        match=(
            "need 70 unspilled DFB indices, exceeding the "
            "(?:64-DFB-index Blackhole|32-DFB-index Wormhole B0) target capacity"
        ),
    ):
        operation(
            lhs,
            first_rhs,
            second_rhs,
            result,
            options="--no-ttl-reuse-user-dfbs",
        )

    final_mlir_path = tmp_path / "external_atom_composition.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    operation(
        lhs,
        first_rhs,
        second_rhs,
        result,
        options="--ttl-reuse-user-dfbs",
    )
    assert (
        _count_final_dfb_allocations(final_mlir_path)
        == EXTERNAL_COMPOSITION_PHYSICAL_DFBS
    )
    _assert_external_composition_result(result, lhs_host, rhs_host, dtype)


# Compiler-managed storage removes the physical-index limit for a mixed
# external and native compute sequence with more than 64 logical DFBs.
@pytest.mark.parametrize(
    ("scratch_operation", "tensor_backed_operation", "dtype"),
    [
        (
            _external_bf16_composition,
            _tensor_backed_bf16_composition,
            torch.bfloat16,
        ),
        (
            _external_f32_composition,
            _tensor_backed_f32_composition,
            torch.float32,
        ),
    ],
    ids=["bf16", "f32"],
)
@pytest.mark.parametrize(
    ("storage_kind", "to_device"),
    [
        ("scratch", to_dram),
        ("scratch", to_l1),
        ("tensor_backed", to_l1_sharded),
    ],
    ids=["scratch-dram", "scratch-l1", "tensor-backed"],
)
@pytest.mark.parametrize(
    "allocation_strategy",
    ["first-fit-decreasing", "best-fit-decreasing"],
    ids=["first-fit", "best-fit"],
)
def test_compiler_l1_external_composition_exceeds_metal_index_limit(
    device,
    scratch_operation,
    tensor_backed_operation,
    dtype,
    storage_kind,
    to_device,
    allocation_strategy,
    reject_metal_dfb_descriptor_creation,
    monkeypatch,
    tmp_path,
):
    reject_metal_dfb_descriptor_creation()
    lhs_host, rhs_host, lhs, first_rhs, second_rhs, result = (
        _make_external_composition_tensors(device, dtype, to_device)
    )
    operation = (
        tensor_backed_operation
        if storage_kind == "tensor_backed"
        else scratch_operation
    )
    final_mlir_path = tmp_path / "compiler_l1_external_composition.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir_path))
    options = (
        "--ttl-memory-model=compiler-l1 "
        "--no-ttl-reuse-user-dfbs "
        f"--ttl-l1-allocation-strategy={allocation_strategy}"
    )

    for _invocation_index in range(2):
        operation(lhs, first_rhs, second_rhs, result, options=options)

    assert (
        _count_final_dfb_allocations(final_mlir_path)
        == EXTERNAL_COMPOSITION_LOGICAL_DFBS
    )
    _assert_external_composition_result(result, lhs_host, rhs_host, dtype)
