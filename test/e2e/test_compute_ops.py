# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal declarative test for all compute operations.

Uses pytest parametrize to test all operations with all configurations.
Single function covers everything - operations and configs are declared as data.
"""

import pytest

from .config_specs import CONFIGS
from .op_specs import COMPUTE_OPS
from .runner import run_compute_test


@pytest.mark.parametrize("op", COMPUTE_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("config", CONFIGS, ids=str)
@pytest.mark.requires_device
def test_compute(op, config, device):
    """Test all single-op compute operations."""
    run_compute_test(op, config, device)
