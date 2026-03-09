# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal declarative test for all compute operations.

Uses pytest parametrize to test all operations with all configurations and dtypes.
Single function covers everything - operations, configs, and dtypes are declared as data.
"""

from dataclasses import replace

import pytest

from .config import get_dtype_ids, get_test_dtypes
from .config_specs import CONFIGS, XFAILS
from .op_specs import COMPUTE_OPS
from .runner import run_compute_test


def _check_xfail(config_str: str, dtype_str: str, op_name: str):
    """Apply xfail marker if (config, dtype, op) matches an XFAILS entry."""
    params = (config_str, dtype_str, op_name)
    for key, reason in XFAILS.items():
        # Pad key with None to length 3 so trailing positions match anything.
        padded = key + (None,) * (3 - len(key))
        if all(k is None or k == p for k, p in zip(padded, params)):
            pytest.xfail(reason)


@pytest.mark.parametrize("op", COMPUTE_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize(
    "config",
    CONFIGS,
    ids=lambda c: str(c),
)
@pytest.mark.parametrize("dtype", get_test_dtypes(), ids=get_dtype_ids())
@pytest.mark.requires_device
def test_compute(op, config, dtype, device):
    """Test all compute operations with all configurations and dtypes."""
    dtype_str = str(dtype).split(".")[-1]
    _check_xfail(str(config), dtype_str, op.name)
    # Create a new config with the specified dtype.
    config_with_dtype = replace(config, dtype=dtype)
    run_compute_test(op, config_with_dtype, device)
