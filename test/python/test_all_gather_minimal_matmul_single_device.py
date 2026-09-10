# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One-device execution coverage for the all-gather matmul example."""

import sys

import pytest

from examples.all_gather_minimal_matmul.__main__ import main

pytestmark = pytest.mark.requires_device


@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
@pytest.mark.parametrize("gather_output", [False, True], ids=["sharded", "replicated"])
def test_single_device(dtype_name, gather_output, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["all_gather_minimal_matmul", "--mesh-shape", "1x1", "--dtype", dtype_name]
        + (["--gather-output"] if gather_output else []),
    )

    main()
