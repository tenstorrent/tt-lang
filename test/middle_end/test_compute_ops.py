# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compute operation tests for middle-end pipeline.

Tests single and fused operations by compiling TTL MLIR through
the full pass pipeline and executing via ttnn.generic_op.
"""

import pytest

from .op_specs import BINARY_OPS, UNARY_OPS, COMPUTE_OPS, FUSED_OPS, ALL_OPS
from .config_specs import CONFIGS, SMOKE_CONFIGS
from .runner import run_compute_test


# All ops (single + fused) with minimal config.
@pytest.mark.parametrize("op", ALL_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("config", SMOKE_CONFIGS, ids=str)
def test_compute_smoke(op, config, device):
    """Smoke test for all operations (single and fused) with minimal config."""
    run_compute_test(op, config, device)


# Binary ops with full configurations.
@pytest.mark.parametrize("op", BINARY_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("config", CONFIGS, ids=str)
def test_binary_ops(op, config, device):
    """Test binary operations across configurations."""
    run_compute_test(op, config, device)


# Unary ops with full configurations.
@pytest.mark.parametrize("op", UNARY_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("config", CONFIGS, ids=str)
def test_unary_ops(op, config, device):
    """Test unary operations across configurations."""
    run_compute_test(op, config, device)


# Fused ops with full configurations.
@pytest.mark.parametrize("op", FUSED_OPS, ids=lambda o: o.name)
@pytest.mark.parametrize("config", CONFIGS, ids=str)
def test_fused_ops(op, config, device):
    """Test fused multi-op kernels across configurations."""
    run_compute_test(op, config, device)

