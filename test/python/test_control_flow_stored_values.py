# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime coverage for tensor values stored across control flow."""

import sys
from pathlib import Path

import pytest
import torch
from ttlang_test_utils import assert_allclose, to_dram

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Inputs.control_flow_stored_values_kernels import (  # noqa: E402
    DFB_FALLBACK_RUNTIME_CASES,
    RUNTIME_CASES,
    SINGLE_BRANCH_RUNTIME_CASES,
    attached_input_stored_value_kernel,
    elif_gap_stored_value_kernel,
    external_use_stored_value_kernel,
    nested_def_stored_value_kernel,
    parent_and_branch_stored_value_kernel,
)

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

DTYPE_TOLERANCES = {
    torch.bfloat16: {"rtol": 1e-2, "atol": 1e-2},
    torch.float32: {"rtol": 1e-3, "atol": 1e-3},
}


def _runtime_input(grid_width, dtype=torch.bfloat16):
    values = torch.arange(32 * grid_width * 32, dtype=torch.float32)
    values = values.reshape(32, grid_width * 32)
    return (values / 1024.0 - 1.0).to(dtype)


def _expected_exp_outputs(input_tensor, output_count):
    return [
        torch.exp(input_tensor[:, output_index * 32 : (output_index + 1) * 32].float())
        for output_index in range(output_count)
    ]


def _source_tile(input_tensor, source_tile_index):
    return input_tensor[
        :,
        source_tile_index * 32 : (source_tile_index + 1) * 32,
    ].float()


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "_case_name,kernel,grid_width,output_count",
    RUNTIME_CASES,
    ids=[case_name for case_name, _kernel, _grid_width, _output_count in RUNTIME_CASES],
)
def test_control_flow_stored_value_runs(
    _case_name, kernel, grid_width, output_count, device, monkeypatch
):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(grid_width)
    input_tensor = to_dram(input_torch, device)
    output_tensors = [
        to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
        for _output_index in range(output_count)
    ]

    kernel(input_tensor, *output_tensors)

    for output_tensor, expected in zip(
        output_tensors, _expected_exp_outputs(input_torch, output_count)
    ):
        actual = ttnn.to_torch(output_tensor).float()
        assert_allclose(actual, expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "_case_name,kernel,grid_width,output_count",
    DFB_FALLBACK_RUNTIME_CASES,
    ids=[
        case_name
        for case_name, _kernel, _grid_width, _output_count in DFB_FALLBACK_RUNTIME_CASES
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_dfb_fallback_stored_values_dtypes(
    _case_name, kernel, grid_width, output_count, dtype, device, monkeypatch
):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(grid_width, dtype=dtype)
    input_tensor = to_dram(input_torch, device)
    output_tensors = [
        to_dram(torch.zeros((32, 32), dtype=dtype), device)
        for _output_index in range(output_count)
    ]

    kernel(input_tensor, *output_tensors)

    for output_tensor, expected in zip(
        output_tensors, _expected_exp_outputs(input_torch, output_count)
    ):
        actual = ttnn.to_torch(output_tensor).float()
        assert_allclose(actual, expected.float(), **DTYPE_TOLERANCES[dtype])


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "_case_name,kernel,grid_width,source_tile_index",
    SINGLE_BRANCH_RUNTIME_CASES,
    ids=[
        case_name
        for case_name, _kernel, _grid_width, _source_tile_index in SINGLE_BRANCH_RUNTIME_CASES
    ],
)
def test_single_branch_store_runs(
    _case_name, kernel, grid_width, source_tile_index, device, monkeypatch
):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(grid_width)
    input_tensor = to_dram(input_torch, device)
    output_tensor = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    kernel(input_tensor, output_tensor)

    expected = torch.exp(_source_tile(input_torch, source_tile_index))
    actual = ttnn.to_torch(output_tensor).float()
    assert_allclose(actual, expected.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.requires_device
def test_elif_gap_stored_value_runs(device, monkeypatch):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(3)
    input_tensor = to_dram(input_torch, device)
    first_output = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    third_output = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    elif_gap_stored_value_kernel(input_tensor, first_output, third_output)

    assert_allclose(
        ttnn.to_torch(first_output).float(),
        torch.exp(_source_tile(input_torch, 0)),
        rtol=1e-2,
        atol=1e-2,
    )
    assert_allclose(
        ttnn.to_torch(third_output).float(),
        torch.exp(_source_tile(input_torch, 2)),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.requires_device
def test_nested_def_stored_value_runs(device, monkeypatch):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(3)
    input_tensor = to_dram(input_torch, device)
    output_tensors = [
        to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
        for _output_index in range(3)
    ]

    nested_def_stored_value_kernel(input_tensor, *output_tensors)

    expected_outputs = [
        torch.exp(_source_tile(input_torch, 0)),
        torch.exp(_source_tile(input_torch, 1)),
        -_source_tile(input_torch, 2),
    ]
    for output_tensor, expected in zip(output_tensors, expected_outputs):
        assert_allclose(
            ttnn.to_torch(output_tensor).float(),
            expected,
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.requires_device
def test_external_use_stored_value_runs(device, monkeypatch):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(2)
    input_tensor = to_dram(input_torch, device)
    first_output = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    second_output = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    side_output = to_dram(torch.zeros((32, 64), dtype=torch.bfloat16), device)

    external_use_stored_value_kernel(
        input_tensor, first_output, second_output, side_output
    )

    expected_full = torch.exp(input_torch.float())
    assert_allclose(
        ttnn.to_torch(first_output).float(),
        expected_full[:, 0:32],
        rtol=1e-2,
        atol=1e-2,
    )
    assert_allclose(
        ttnn.to_torch(second_output).float(),
        expected_full[:, 32:64],
        rtol=1e-2,
        atol=1e-2,
    )
    assert_allclose(
        ttnn.to_torch(side_output).float(),
        -expected_full,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.requires_device
def test_parent_and_branch_stored_value_runs(device, monkeypatch):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(2)
    input_tensor = to_dram(input_torch, device)
    always_output = to_dram(torch.zeros((32, 64), dtype=torch.bfloat16), device)
    branch_output = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    parent_and_branch_stored_value_kernel(input_tensor, always_output, branch_output)

    expected_full = torch.exp(input_torch.float())
    assert_allclose(
        ttnn.to_torch(always_output).float(),
        expected_full,
        rtol=1e-2,
        atol=1e-2,
    )
    assert_allclose(
        ttnn.to_torch(branch_output).float(),
        expected_full[:, 0:32],
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.requires_device
def test_attached_input_stored_value_runs(device, monkeypatch):
    monkeypatch.delenv("TTLANG_COMPILE_ONLY", raising=False)
    input_torch = _runtime_input(2)
    input_tensor = to_dram(input_torch, device)
    first_output = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    second_output = to_dram(torch.zeros((32, 32), dtype=torch.bfloat16), device)

    attached_input_stored_value_kernel(input_tensor, first_output, second_output)

    assert_allclose(
        ttnn.to_torch(first_output).float(),
        _source_tile(input_torch, 0),
        rtol=0,
        atol=0,
    )
    assert_allclose(
        ttnn.to_torch(second_output).float(),
        _source_tile(input_torch, 1),
        rtol=0,
        atol=0,
    )
