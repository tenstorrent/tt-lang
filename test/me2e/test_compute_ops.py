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
from .config_specs import CONFIGS
from .op_specs import COMPUTE_OPS
from .runner import run_compute_test


@pytest.mark.parametrize("op", COMPUTE_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize(
    "config",
    CONFIGS,
    ids=lambda c: f"{c.block_h}x{c.block_w}_buf{c.buffer_factor}_{c.memory_layout.value}",
)
@pytest.mark.parametrize("dtype", get_test_dtypes(), ids=get_dtype_ids())
@pytest.mark.requires_device
def test_compute(op, config, dtype, device):
    """Test all compute operations with all configurations and dtypes."""
    # Create a new config with the specified dtype.
    config_with_dtype = replace(config, dtype=dtype)
    run_compute_test(op, config_with_dtype, device)
