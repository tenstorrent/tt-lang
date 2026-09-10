# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One-device execution coverage for the all-gather matmul example."""

import sys

import pytest

from examples.all_gather_minimal_matmul.__main__ import main

pytestmark = pytest.mark.requires_device


@pytest.mark.parametrize("dtype_name", ["bf16", "fp32"])
@pytest.mark.parametrize("algorithm", ["all_to_all", "ring"])
@pytest.mark.parametrize(
    "variant,gather_output",
    [("n_sharded", False), ("n_sharded", True), ("replicated", False)],
    ids=["n-sharded", "output-gather", "replicated-weights"],
)
def test_single_device(dtype_name, variant, gather_output, algorithm, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["all_gather_minimal_matmul", "--mesh-shape", "1x1", "--dtype", dtype_name]
        + ["--activation-all-gather", algorithm, "--output-all-gather", algorithm]
        + (["--gather-output"] if gather_output else []),
    )

    main(variant=variant)
