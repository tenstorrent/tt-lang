# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for E2E test utilities.

Tests ULP computation and comparison functions without requiring hardware.
"""

import torch
import pytest

from .utils import ulp, get_default_ulp_threshold, compare_tensors


class TestULP:
    """Test ULP computation."""

    def test_ulp_float32(self):
        """Test ULP for float32."""
        x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
        ulp_vals = ulp(x)
        assert ulp_vals.dtype == torch.float32
        assert torch.all(ulp_vals > 0)

    def test_ulp_bfloat16(self):
        """Test ULP for bfloat16."""
        x = torch.tensor([1.0, 2.0, 0.5], dtype=torch.bfloat16)
        ulp_vals = ulp(x)
        assert ulp_vals.dtype == torch.bfloat16
        assert torch.all(ulp_vals > 0)

    def test_ulp_symmetry(self):
        """Test ULP(-x) == ULP(x) for same magnitude."""
        x = torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.float32)
        ulp_vals = ulp(x)
        # ULP should be same for positive and negative of same magnitude
        assert torch.allclose(ulp_vals[0], ulp_vals[1])  # 1.0 vs -1.0
        assert torch.allclose(ulp_vals[2], ulp_vals[3])  # 2.0 vs -2.0


class TestDefaultThresholds:
    """Test default ULP thresholds."""

    def test_bfloat16_threshold(self):
        """BF16 should have 2 ULP threshold."""
        assert get_default_ulp_threshold(torch.bfloat16) == 2.0

    def test_float32_threshold(self):
        """FP32 should have 1 ULP threshold."""
        assert get_default_ulp_threshold(torch.float32) == 1.0

    def test_int32_threshold(self):
        """INT32 should have 0 ULP (exact)."""
        assert get_default_ulp_threshold(torch.int32) == 0.0


class TestCompareTensors:
    """Test tensor comparison."""

    def test_exact_match(self):
        """Exact match should pass."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = compare_tensors(x, x)
        assert result.passed
        assert result.max_ulp == 0.0

    def test_small_difference(self):
        """Small difference within threshold should pass."""
        x = torch.tensor([1.0], dtype=torch.float32)
        y = torch.nextafter(x, torch.tensor(float("inf")))
        result = compare_tensors(x, y, ulp_threshold=1.0)
        assert result.passed
        assert result.max_ulp <= 1.0

    def test_large_difference(self):
        """Large difference should fail."""
        x = torch.tensor([1.0], dtype=torch.float32)
        y = torch.tensor([2.0], dtype=torch.float32)
        result = compare_tensors(x, y, ulp_threshold=1.0)
        assert not result.passed
        assert result.max_ulp > 1.0

    def test_empty_tensors(self):
        """Empty tensors should match."""
        x = torch.tensor([], dtype=torch.float32)
        y = torch.tensor([], dtype=torch.float32)
        result = compare_tensors(x, y)
        assert result.passed

    def test_bfloat16_comparison(self):
        """BF16 comparison with auto threshold."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        y = x.clone()
        result = compare_tensors(x, y)
        assert result.passed
