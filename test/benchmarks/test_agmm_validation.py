# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from benchmarks.all_gather_minimal_matmul.__main__ import validate_output


def test_bf16_validation_uses_dtype_appropriate_absolute_tolerance():
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    actual = expected.clone()
    actual[0, 0] = 0.2

    validation = validate_output(actual, expected, "bf16")

    assert validation["rtol"] == 0.05
    assert validation["atol"] == 1.0


def test_fp32_validation_retains_tight_absolute_tolerance():
    expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    actual = expected.clone()
    actual[0, 0] = 0.01

    with pytest.raises(AssertionError, match="Tensor comparison failed"):
        validate_output(actual, expected, "fp32")
