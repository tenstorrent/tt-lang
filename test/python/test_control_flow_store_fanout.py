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

from Inputs.control_flow_store_fanout_kernels import RUNTIME_CASES  # noqa: E402

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)


def _runtime_input(grid_width):
    values = torch.arange(32 * grid_width * 32, dtype=torch.float32)
    values = values.reshape(32, grid_width * 32)
    return (values / 1024.0 - 1.0).to(torch.bfloat16)


def _expected_exp_outputs(input_tensor, output_count):
    return [
        torch.exp(input_tensor[:, output_index * 32 : (output_index + 1) * 32].float())
        for output_index in range(output_count)
    ]


@pytest.mark.requires_device
@pytest.mark.parametrize(
    "_case_name,kernel,grid_width,output_count",
    RUNTIME_CASES,
    ids=[case_name for case_name, _kernel, _grid_width, _output_count in RUNTIME_CASES],
)
def test_control_flow_store_fanout_runs(
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
        assert_allclose(actual, expected.float(), rtol=0.05, atol=0.5)
