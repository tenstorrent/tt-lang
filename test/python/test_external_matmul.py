# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device coverage for external matmul with compiler-managed L1 storage.

The focused cases validate one external matmul with compiler-owned and
tensor-backed storage, reset, and reconfiguration. The model-like case composes
external matmuls with native normalization, activation, and residual operations.
"""

import os
import re
from functools import partial

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

import ttl  # noqa: E402
from ttl import ttl_api  # noqa: E402
from ttlang_test_utils import to_dram, to_l1, to_l1_sharded  # noqa: E402
from utils.correctness import assert_allclose  # noqa: E402

pytestmark = pytest.mark.requires_device

TILE = 32
MATMUL_DIMENSIONS = [(1, 1, 1), (1, 2, 1), (2, 2, 2)]
EXTERNAL_MATMUL_HEADER = os.path.join(
    os.path.dirname(__file__), "include", "external_matmul.hpp"
)
STORAGE_OPTIONS = [
    pytest.param("--ttl-memory-model=metal-cb", id="metal"),
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


def _make_matmul_references(dimensions, dtype):
    rows, inner, columns = dimensions
    lhs_indices = torch.arange(rows * TILE * inner * TILE).reshape(
        rows * TILE, inner * TILE
    )
    rhs_indices = torch.arange(inner * TILE * columns * TILE).reshape(
        inner * TILE, columns * TILE
    )
    lhs = (lhs_indices.remainder(3) - 1).to(dtype)
    rhs = ((2 * rhs_indices + 1).remainder(3) - 1).to(dtype)
    return lhs, rhs


def _make_external_matmul_atom(rows, inner, columns):
    lhs_pages = rows * inner
    rhs_pages = inner * columns
    result_pages = rows * columns

    @ttl.operation()
    def external_matmul(lhs: ttl.DFB, rhs: ttl.DFB, result: ttl.DFB):
        ttl.call_extern_func(
            EXTERNAL_MATMUL_HEADER,
            "ttl_external_matmul",
            template_args=[
                ttl.dfb_descriptor(lhs),
                ttl.dfb_descriptor(rhs),
                ttl.dfb_descriptor(result),
                rows,
                inner,
                columns,
            ],
            dfb_effects=[
                ttl.DFBEffect.reserve(result, tiles=result_pages),
                ttl.DFBEffect.wait(lhs, tiles=lhs_pages),
                ttl.DFBEffect.wait(rhs, tiles=rhs_pages),
                ttl.DFBEffect.push(result, tiles=result_pages),
                ttl.DFBEffect.pop(lhs, tiles=lhs_pages),
                ttl.DFBEffect.pop(rhs, tiles=rhs_pages),
            ],
            kernel=ttl.KernelKind.COMPUTE,
        )

    return external_matmul


def _make_external_matmul_operation(data_format, tensor_backed, dimensions):
    rows, inner, columns = dimensions
    external_matmul = _make_external_matmul_atom(rows, inner, columns)
    if tensor_backed:

        @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
        def external_matmul_operation(lhs, rhs, result):
            lhs_dfb = ttl.make_tensor_backed_dfb(lhs, shape=(rows, inner))
            rhs_dfb = ttl.make_tensor_backed_dfb(rhs, shape=(inner, columns))
            result_dfb = ttl.make_tensor_backed_dfb(result, shape=(rows, columns))
            lhs_dfb.publish()
            rhs_dfb.publish()
            external_matmul(lhs_dfb, rhs_dfb, result_dfb)
            result_source = result_dfb.wait()
            result_source.pop(kernel=ttl.KernelKind.DATA_MOVEMENT)

    else:

        @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
        def external_matmul_operation(lhs, rhs, result):
            lhs_dfb = ttl.make_dfb(data_format, shape=(rows, inner), block_count=2)
            rhs_dfb = ttl.make_dfb(data_format, shape=(inner, columns), block_count=2)
            result_dfb = ttl.make_dfb(data_format, shape=(rows, columns), block_count=2)
            with lhs_dfb.reserve() as lhs_destination:
                ttl.copy(lhs[0:rows, 0:inner], lhs_destination).wait()
            with rhs_dfb.reserve() as rhs_destination:
                ttl.copy(rhs[0:inner, 0:columns], rhs_destination).wait()
            external_matmul(lhs_dfb, rhs_dfb, result_dfb)
            with result_dfb.wait() as result_source:
                ttl.copy(result_source, result[0:rows, 0:columns]).wait()

    return external_matmul_operation


def _make_external_gated_mlp(data_format):
    external_matmul = _make_external_matmul_atom(1, 1, 1)

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
    def external_gated_mlp(source, gate_weight, up_weight, down_weight, output):
        input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        residual_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        gate_input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        up_input_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        gate_weight_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        up_weight_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        down_weight_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=1)
        gate_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        up_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        activation_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        projection_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)
        output_dfb = ttl.make_dfb(data_format, shape=(1, 1), block_count=2)

        with input_dfb.reserve() as destination:
            ttl.copy(source[0, 0], destination).wait()
        with residual_dfb.reserve() as destination:
            ttl.copy(source[0, 0], destination).wait()
        with gate_weight_dfb.reserve() as destination:
            ttl.copy(gate_weight[0, 0], destination).wait()
        with up_weight_dfb.reserve() as destination:
            ttl.copy(up_weight[0, 0], destination).wait()
        with down_weight_dfb.reserve() as destination:
            ttl.copy(down_weight[0, 0], destination).wait()

        with (
            input_dfb.wait() as input_block,
            gate_input_dfb.reserve() as gate_input,
            up_input_dfb.reserve() as up_input,
        ):
            squared = input_block * input_block
            total = ttl.math.reduce_sum(squared, dims=[1], shape=(1, 1))
            biased = total * 0.03125 + ttl.block.fill(
                0.00001, shape=(1, 1), dtype=input_block.dtype
            )
            inverse = ttl.math.rsqrt(biased)
            normalized = input_block * ttl.block.broadcast(
                inverse, dims=[1], shape=(1, 1)
            )
            gate_input.store(normalized)
            up_input.store(normalized)

        external_matmul(gate_input_dfb, gate_weight_dfb, gate_dfb)
        external_matmul(up_input_dfb, up_weight_dfb, up_dfb)

        with (
            gate_dfb.wait() as gate_block,
            up_dfb.wait() as up_block,
            activation_dfb.reserve() as activation_block,
        ):
            limited_gate = 4.0 * ttl.math.tanh(gate_block * 0.25)
            limited_up = 25.0 * ttl.math.tanh(up_block * 0.04)
            activation_block.store(
                limited_gate * ttl.math.sigmoid(gate_block) * limited_up
            )

        external_matmul(activation_dfb, down_weight_dfb, projection_dfb)

        with (
            projection_dfb.wait() as projection,
            residual_dfb.wait() as residual,
            output_dfb.reserve() as output_block,
        ):
            output_block.store(projection + residual)

        with output_dfb.wait() as output_source:
            ttl.copy(output_source, output[0, 0]).wait()

    return external_gated_mlp


def _make_external_matmul_reset(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reset = ttl.DFBReset(
        participants=(compute_kernel, reader_kernel, writer_kernel),
    )

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
    def external_matmul_reset(lhs, rhs, replacement, result):
        lhs_dfb = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)
        rhs_dfb = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)
        result_storage = ttl.make_dfb_allocation_group()
        stale_result = ttl.make_dfb(
            data_format,
            shape=(2, 2),
            block_count=2,
            allocation_group=result_storage,
        )
        current_result = ttl.make_dfb(
            data_format,
            shape=(2, 2),
            block_count=2,
            allocation_group=result_storage,
        )

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                EXTERNAL_MATMUL_HEADER,
                "ttl_external_matmul",
                template_args=[
                    ttl.dfb_descriptor(lhs_dfb),
                    ttl.dfb_descriptor(rhs_dfb),
                    ttl.dfb_descriptor(stale_result),
                    2,
                    2,
                    2,
                ],
                dfb_effects=[
                    ttl.DFBEffect.reserve(stale_result, tiles=4),
                    ttl.DFBEffect.wait(lhs_dfb, tiles=4),
                    ttl.DFBEffect.wait(rhs_dfb, tiles=4),
                    ttl.DFBEffect.push(stale_result, tiles=4),
                    ttl.DFBEffect.pop(lhs_dfb, tiles=4),
                    ttl.DFBEffect.pop(rhs_dfb, tiles=4),
                ],
            )
            ttl.reset_dfbs(reset, dfbs=[stale_result])

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with lhs_dfb.reserve() as destination:
                ttl.copy(lhs[0:2, 0:2], destination).wait()
            with rhs_dfb.reserve() as destination:
                ttl.copy(rhs[0:2, 0:2], destination).wait()
            ttl.reset_dfbs(reset, dfbs=[stale_result])
            with current_result.reserve() as destination:
                ttl.copy(replacement[0:2, 0:2], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            ttl.reset_dfbs(reset, dfbs=[stale_result])
            with current_result.wait() as source:
                ttl.copy(source, result[0:2, 0:2]).wait()

    return external_matmul_reset


def _make_external_matmul_reconfiguration(data_format):
    compute_kernel = ttl.Kernel(ttl.KernelKind.COMPUTE)
    reader_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    writer_kernel = ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT)
    reconfiguration = ttl.DFBReconfiguration(
        participants=(compute_kernel, reader_kernel, writer_kernel),
        discard_dfb_state=True,
    )

    @ttl.operation(grid=(1, 1), fp32_dest_acc_en=data_format == "float32")
    def external_matmul_reconfiguration(lhs, rhs, result):
        before_lhs = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)
        before_rhs = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)
        before_result = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)
        after_lhs = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)
        after_rhs = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)
        after_result = ttl.make_dfb(data_format, shape=(2, 2), block_count=2)

        @ttl.compute(kernel=compute_kernel)
        def compute():
            ttl.call_extern_func(
                EXTERNAL_MATMUL_HEADER,
                "ttl_external_matmul",
                template_args=[
                    ttl.dfb_descriptor(before_lhs),
                    ttl.dfb_descriptor(before_rhs),
                    ttl.dfb_descriptor(before_result),
                    2,
                    2,
                    2,
                ],
                dfb_effects=[
                    ttl.DFBEffect.reserve(before_result, tiles=4),
                    ttl.DFBEffect.wait(before_lhs, tiles=4),
                    ttl.DFBEffect.wait(before_rhs, tiles=4),
                    ttl.DFBEffect.push(before_result, tiles=4),
                    ttl.DFBEffect.pop(before_lhs, tiles=4),
                    ttl.DFBEffect.pop(before_rhs, tiles=4),
                ],
            )
            ttl.reconfigure_dfbs(reconfiguration)
            ttl.call_extern_func(
                EXTERNAL_MATMUL_HEADER,
                "ttl_external_matmul",
                template_args=[
                    ttl.dfb_descriptor(after_lhs),
                    ttl.dfb_descriptor(after_rhs),
                    ttl.dfb_descriptor(after_result),
                    2,
                    2,
                    2,
                ],
                dfb_effects=[
                    ttl.DFBEffect.reserve(after_result, tiles=4),
                    ttl.DFBEffect.wait(after_lhs, tiles=4),
                    ttl.DFBEffect.wait(after_rhs, tiles=4),
                    ttl.DFBEffect.push(after_result, tiles=4),
                    ttl.DFBEffect.pop(after_lhs, tiles=4),
                    ttl.DFBEffect.pop(after_rhs, tiles=4),
                ],
            )

        @ttl.datamovement(kernel=reader_kernel)
        def read():
            with before_lhs.reserve() as destination:
                ttl.copy(lhs[0:2, 0:2], destination).wait()
            with before_rhs.reserve() as destination:
                ttl.copy(rhs[0:2, 0:2], destination).wait()
            ttl.reconfigure_dfbs(reconfiguration)
            with after_lhs.reserve() as destination:
                ttl.copy(lhs[0:2, 0:2], destination).wait()
            with after_rhs.reserve() as destination:
                ttl.copy(rhs[0:2, 0:2], destination).wait()

        @ttl.datamovement(kernel=writer_kernel)
        def write():
            with before_result.wait() as source:
                ttl.copy(source, result[0:2, 0:2]).wait()
            ttl.reconfigure_dfbs(reconfiguration)
            with after_result.wait() as source:
                ttl.copy(source, result[2:4, 0:2]).wait()

    return external_matmul_reconfiguration


EXTERNAL_GATED_MLP_OPERATIONS = {
    "bf16": _make_external_gated_mlp("bf16"),
    "float32": _make_external_gated_mlp("float32"),
}
EXTERNAL_MATMUL_RESET_OPERATIONS = {
    "bf16": _make_external_matmul_reset("bf16"),
    "float32": _make_external_matmul_reset("float32"),
}
EXTERNAL_MATMUL_RECONFIGURATION_OPERATIONS = {
    "bf16": _make_external_matmul_reconfiguration("bf16"),
    "float32": _make_external_matmul_reconfiguration("float32"),
}


# Validates external matmul across storage backends and ownership forms.
@pytest.mark.parametrize(
    ("data_format", "dtype"),
    [("bf16", torch.bfloat16), ("float32", torch.float32)],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize(
    ("tensor_backed", "to_device"),
    [
        (False, to_dram),
        (False, to_l1),
        (True, partial(to_l1_sharded, layout="height")),
        (True, partial(to_l1_sharded, layout="width")),
        (True, partial(to_l1_sharded, layout="block")),
    ],
    ids=[
        "scratch-dram",
        "scratch-l1",
        "tensor-backed-height",
        "tensor-backed-width",
        "tensor-backed-block",
    ],
)
@pytest.mark.parametrize("storage_options", STORAGE_OPTIONS)
@pytest.mark.parametrize(
    "dimensions",
    MATMUL_DIMENSIONS,
    ids=["one-tile", "inner-two", "two-by-two"],
)
def test_external_matmul(
    device,
    data_format,
    dtype,
    tensor_backed,
    to_device,
    storage_options,
    dimensions,
    reject_metal_dfb_descriptor_creation,
):
    rows, inner, columns = dimensions
    reference_lhs, reference_rhs = _make_matmul_references(dimensions, dtype)
    lhs = to_device(reference_lhs, device)
    rhs = to_device(reference_rhs, device)
    result = to_device(torch.zeros((rows * TILE, columns * TILE), dtype=dtype), device)
    operation = _make_external_matmul_operation(data_format, tensor_backed, dimensions)
    if "--ttl-memory-model=compiler-l1" in storage_options:
        reject_metal_dfb_descriptor_creation()

    for _invocation_index in range(2):
        operation(lhs, rhs, result, options=storage_options)

    expected = reference_lhs.float() @ reference_rhs.float()
    assert_allclose(ttnn.to_torch(result).float(), expected, rtol=0, atol=0)


# Validates synchronized reset after a complete multi-page external transaction.
@pytest.mark.parametrize(
    ("data_format", "dtype"),
    [("bf16", torch.bfloat16), ("float32", torch.float32)],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize("storage_options", STORAGE_OPTIONS)
def test_external_matmul_across_reset(
    device,
    data_format,
    dtype,
    to_device,
    storage_options,
    reject_metal_dfb_descriptor_creation,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reset support")

    reference_lhs, reference_rhs = _make_matmul_references((2, 2, 2), dtype)
    element_indices = torch.arange(2 * TILE * 2 * TILE).reshape(2 * TILE, 2 * TILE)
    replacement_reference = ((element_indices.remainder(17) - 8) / 16).to(dtype)
    lhs = to_device(reference_lhs, device)
    rhs = to_device(reference_rhs, device)
    replacement = to_device(replacement_reference, device)
    result = to_device(torch.zeros_like(replacement_reference), device)
    operation = EXTERNAL_MATMUL_RESET_OPERATIONS[data_format]
    if "--ttl-memory-model=compiler-l1" in storage_options:
        reject_metal_dfb_descriptor_creation()

    options = f"--ttl-reuse-user-dfbs --ttl-specialize-cores {storage_options}"
    for _invocation_index in range(2):
        operation(lhs, rhs, replacement, result, options=options)

    tolerance = 0.05 if dtype == torch.bfloat16 else 1e-6
    assert_allclose(
        ttnn.to_torch(result).float(),
        replacement_reference.float(),
        rtol=tolerance,
        atol=tolerance,
    )


# Validates matmul completion and payload reuse across reconfiguration.
@pytest.mark.parametrize(
    ("data_format", "dtype"),
    [("bf16", torch.bfloat16), ("float32", torch.float32)],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "allocation_strategy",
    ["first-fit-decreasing", "best-fit-decreasing"],
    ids=["first-fit", "best-fit"],
)
def test_external_matmul_across_reconfiguration(
    device,
    data_format,
    dtype,
    to_device,
    allocation_strategy,
    reject_metal_dfb_descriptor_creation,
    monkeypatch,
    tmp_path,
):
    if ttl_api._detect_device_arch(device) != "blackhole":
        pytest.skip("requires Blackhole DFB reconfiguration support")

    reference_lhs, reference_rhs = _make_matmul_references((2, 2, 2), dtype)
    lhs = to_device(reference_lhs, device)
    rhs = to_device(reference_rhs, device)
    result = to_device(torch.zeros((4 * TILE, 2 * TILE), dtype=dtype), device)
    operation = EXTERNAL_MATMUL_RECONFIGURATION_OPERATIONS[data_format]
    final_mlir = tmp_path / "external_matmul_reconfiguration.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))
    reject_metal_dfb_descriptor_creation()
    options = (
        "--ttl-memory-model=compiler-l1 "
        f"--ttl-l1-allocation-strategy={allocation_strategy}"
    )

    for _invocation_index in range(2):
        operation(lhs, rhs, result, options=options)

    expected_block = reference_lhs.float() @ reference_rhs.float()
    expected = torch.cat((expected_block, expected_block), dim=0)
    assert_allclose(ttnn.to_torch(result).float(), expected, rtol=0, atol=0)
    final_ir = final_mlir.read_text()
    payload_offsets = [
        int(offset) for offset in re.findall(r"l1_payload_offset = (\d+)", final_ir)
    ]
    assert len(payload_offsets) == 6
    assert len(set(payload_offsets)) == 3


# Validates explicit external DFB effects in a model-like compute sequence.
@pytest.mark.parametrize(
    ("data_format", "dtype"),
    [("bf16", torch.bfloat16), ("float32", torch.float32)],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize("to_device", [to_dram, to_l1], ids=["dram", "l1"])
@pytest.mark.parametrize(
    "allocation_strategy",
    ["first-fit-decreasing", "best-fit-decreasing"],
    ids=["first-fit", "best-fit"],
)
def test_external_matmul_in_gated_mlp(
    device,
    data_format,
    dtype,
    to_device,
    allocation_strategy,
    reject_metal_dfb_descriptor_creation,
    monkeypatch,
    tmp_path,
):
    element_indices = torch.arange(TILE * TILE).reshape(TILE, TILE)
    reference = ((element_indices.remainder(41) - 20) / 64).to(dtype)
    weights = [
        (((factor * element_indices + factor).remainder(37) - 18) / 128).to(dtype)
        for factor in (1, 3, 5)
    ]
    normalized = (
        reference.float()
        * torch.rsqrt(reference.float().square().mean(dim=1, keepdim=True) + 0.00001)
    ).to(dtype)
    gate = (normalized.float() @ weights[0].float()).to(dtype)
    up = (normalized.float() @ weights[1].float()).to(dtype)
    activation = (
        4.0
        * torch.tanh(gate.float() * 0.25)
        * torch.sigmoid(gate.float())
        * 25.0
        * torch.tanh(up.float() * 0.04)
    ).to(dtype)
    expected = (activation.float() @ weights[2].float() + reference.float()).to(dtype)
    source = to_device(reference, device)
    device_weights = [to_device(weight, device) for weight in weights]
    output = to_device(torch.zeros_like(reference), device)
    operation = EXTERNAL_GATED_MLP_OPERATIONS[data_format]
    final_mlir = tmp_path / "external_gated_mlp.mlir"
    monkeypatch.setenv("TTLANG_FINAL_MLIR", str(final_mlir))
    reject_metal_dfb_descriptor_creation()
    options = (
        "--ttl-memory-model=compiler-l1 --ttl-reuse-user-dfbs "
        f"--ttl-l1-allocation-strategy={allocation_strategy}"
    )

    for _invocation_index in range(2):
        operation(source, *device_weights, output, options=options)

    tolerance = 0.04 if dtype == torch.bfloat16 else 0.004
    assert_allclose(
        ttnn.to_torch(output).float(),
        expected.float(),
        rtol=tolerance,
        atol=tolerance,
    )
    final_ir = final_mlir.read_text()
    payload_offsets = [
        int(offset) for offset in re.findall(r"l1_payload_offset = (\d+)", final_ir)
    ]
    allocation_bytes = [
        int(size) for size in re.findall(r"l1_allocation_bytes = (\d+)", final_ir)
    ]
    arena_bytes = int(re.search(r"ttl.l1_arena_bytes = (\d+)", final_ir).group(1))
    assert len(set(payload_offsets)) < len(payload_offsets)
    assert arena_bytes < sum(allocation_bytes)
