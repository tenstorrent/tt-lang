# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest coverage for computed receiver-address PipeNet correctness."""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from computed_address_protocol import (  # noqa: E402
    ALL_GATHER_WIDTH,
    TILE,
    _assert_copy_matches,
    _make_input,
    point_to_point_computed_address,
    row_all_gather_computed_address,
)
from ttlang_test_utils import to_dram  # noqa: E402


def _run_copy_kernel(device, dtype, shape, kernel):
    inp_torch = _make_input(shape, dtype)
    out_torch = torch.zeros(shape, dtype=dtype)
    inp = to_dram(inp_torch, device)
    out = to_dram(out_torch, device)

    kernel(inp, out)
    ttnn.synchronize_device(device)
    _assert_copy_matches(inp_torch, ttnn.to_torch(out), dtype)


@pytest.mark.parametrize(
    ("dtype", "shape", "kernel"),
    [
        pytest.param(
            torch.bfloat16,
            (TILE, TILE),
            point_to_point_computed_address,
            id="p2p-bf16",
        ),
        pytest.param(
            torch.float32,
            (TILE, TILE),
            point_to_point_computed_address,
            id="p2p-fp32",
        ),
        pytest.param(
            torch.bfloat16,
            (TILE, ALL_GATHER_WIDTH * TILE),
            row_all_gather_computed_address,
            id="all-gather-bf16",
        ),
        pytest.param(
            torch.float32,
            (TILE, ALL_GATHER_WIDTH * TILE),
            row_all_gather_computed_address,
            id="all-gather-fp32",
        ),
    ],
)
def test_computed_address_protocol(device, dtype, shape, kernel):
    _run_copy_kernel(device, dtype, shape, kernel)
