# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ME2E tests for elementwise compare operations (ttl.gt, ttl.lt).

Lowering uses sub_binary_tile + gtz_tile / ltz_tile on DST. Golden references use
torch.gt / torch.lt; bool tensors are converted to float 0/1 for device I/O.
Validation requires binary-identical masks (torch.equal), not PCC/ULP.

Run only these tests:

    pytest -v test/me2e/test_compare_ops.py

Class-based coverage also appears in ops/test_binary.py as TestGt* / TestLt*.
"""

from dataclasses import replace

import pytest

from .config import get_dtype_ids, get_test_dtypes
from .config_specs import CONFIGS, XFAILS
from .op_specs import COMPUTE_OPS
from .runner import run_compute_test

_COMPARE_OPS = tuple(op for op in COMPUTE_OPS if op.name in ("gt", "lt"))
assert len(_COMPARE_OPS) == 2, (
    "COMPUTE_OPS must include exactly gt and lt; check ELEMENTWISE_OPS / OP_TORCH_MAP "
    f"in test/me2e/ops/__init__.py (got { [o.name for o in _COMPARE_OPS]!r})"
)


def _check_xfail(config_str: str, dtype_str: str, op_name: str):
    """Apply xfail marker if (config, dtype, op) matches an XFAILS entry."""
    params = (config_str, dtype_str, op_name)
    for key, reason in XFAILS.items():
        padded = key + (None,) * (3 - len(key))
        if all(k is None or k == p for k, p in zip(padded, params)):
            pytest.xfail(reason)


@pytest.mark.parametrize("op", _COMPARE_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize(
    "config",
    CONFIGS,
    ids=lambda c: str(c),
)
@pytest.mark.parametrize("dtype", get_test_dtypes(), ids=get_dtype_ids())
@pytest.mark.requires_device
def test_compare(op, config, dtype, device):
    """Elementwise gt/lt: same harness as test_compute, scoped to compare ops."""
    dtype_str = str(dtype).split(".")[-1]
    _check_xfail(str(config), dtype_str, op.name)
    config_with_dtype = replace(config, dtype=dtype)
    run_compute_test(op, config_with_dtype, device)
