# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ME2E tests for elementwise compare operations.

Lowering uses TTKernel's native binary SFPU compare ops. Golden references use
the matching torch comparison ops; bool tensors are converted to float 0/1 for
device I/O.
Validation requires binary-identical masks (torch.equal), not PCC/ULP.

Parametrization applies the test dtype to TestConfig before stringifying, so ids
look like ``float32-2x2_float32_buf2_interleaved-gt`` (tensor dtype is uniform;
there is no separate lhs/rhs dtype). Older patterns such as
``float32-2x2_bfloat16_...`` came from ``str(config)`` using the template default
dtype while the first segment was the parametrized dtype — misleading, not mixed
precision.

BF16 compare cases are skipped (exact mask vs golden). Float32 1x1 ``lt`` /
``eq`` / ``ne`` are skipped (exact mask mismatch vs torch golden on that grid;
``gt`` still runs).

Run only these tests:

    pytest -v test/me2e/test_compare_ops.py

Class-based coverage also appears in ops/test_binary.py as TestGt* / TestLt* /
TestEq* / TestNe* (BF16 class tests omitted there).
"""

import itertools
from dataclasses import replace

import pytest
import torch

from .config import get_test_dtypes
from .config_specs import CONFIGS, XFAILS
from .op_specs import COMPUTE_OPS
from .runner import run_compute_test

_COMPARE_OP_NAMES = frozenset(("eq", "ne", "gt", "lt"))
_COMPARE_OPS = tuple(op for op in COMPUTE_OPS if op.name in _COMPARE_OP_NAMES)
assert len(_COMPARE_OPS) == len(_COMPARE_OP_NAMES), (
    "COMPUTE_OPS must include exactly eq/ne/gt/lt; check ELEMENTWISE_OPS / OP_TORCH_MAP "
    f"in test/me2e/ops/__init__.py (got { [o.name for o in _COMPARE_OPS]!r})"
)


def _check_xfail(config_str: str, dtype_str: str, op_name: str):
    """Apply xfail marker if (config, dtype, op) matches an XFAILS entry."""
    params = (config_str, dtype_str, op_name)
    for key, reason in XFAILS.items():
        padded = key + (None,) * (3 - len(key))
        if all(k is None or k == p for k, p in zip(padded, params)):
            pytest.xfail(reason)


def _compare_test_params():
    """(op, config) with dtype applied; skip BF16; ids include real dtype in config segment."""
    params = []
    for op, config, dtype in itertools.product(
        _COMPARE_OPS, CONFIGS, get_test_dtypes()
    ):
        cfg = replace(config, dtype=dtype)
        dtype_label = str(dtype).split(".")[-1]
        test_id = f"{dtype_label}-{cfg}-{op.name}"
        skip_bf16 = pytest.mark.skipif(
            # TODO: remove this once we have BF16 support for compare ops
            dtype == torch.bfloat16,
            reason=("Compare ME2E: skip BF16 (exact 0/1 mask vs golden)."),
        )
        skip_f32_1x1_compare = pytest.mark.skipif(
            # TODO: investigate and fix this, float32 should work
            dtype == torch.float32
            and cfg.block_h == 1
            and cfg.block_w == 1
            and op.name in ("lt", "eq", "ne"),
            reason=(
                "Compare ME2E: skip float32 1x1 lt/eq/ne "
                "(exact mask vs torch golden)."
            ),
        )
        params.append(
            pytest.param(op, cfg, id=test_id, marks=(skip_bf16, skip_f32_1x1_compare))
        )
    return params


@pytest.mark.parametrize("op,config", _compare_test_params())
@pytest.mark.requires_device
def test_compare(op, config, device):
    """Elementwise compare ops: same harness as test_compute, scoped here."""
    dtype_str = str(config.dtype).split(".")[-1]
    _check_xfail(str(config), dtype_str, op.name)
    run_compute_test(op, config, device)
