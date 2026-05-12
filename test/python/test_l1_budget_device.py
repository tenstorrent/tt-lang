# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for L1 budget validation with device-aware budget detection.

Verifies that build_cb_descriptors correctly rejects circular buffer
allocations that exceed the available L1 when:
  1. The device is opened with a reduced worker_l1_size.
  2. L1 tensors are allocated on the device (consuming L1 on core 0,0).
  3. Both reduced worker_l1_size and L1 tensor allocations are combined.

Each scenario self-calibrates by querying the device for actual remaining
L1, then creates a CB allocation slightly above that limit. This keeps the
tests hardware-independent (they work on any chip with any L1 geometry).
"""

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttl.dataflow_buffer import CompilerAllocatedDFBConfig
from ttl.constants import DEFAULT_L1_CB_BUDGET_BYTES
from ttl.kernel_runner import build_cb_descriptors, get_min_remaining_l1_for_device

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ttlang_test_utils import is_hardware_available, to_l1, to_l1_sharded

pytestmark = pytest.mark.skipif(
    not is_hardware_available(), reason="No Tenstorrent device available"
)

TILE_BYTES = 2048  # bf16 tile: 32 * 32 * 2


def _overflow_config(remaining_bytes):
    """Build a single CompilerAllocatedDFBConfig whose size exceeds *remaining_bytes*."""
    overflow_tiles = (remaining_bytes // TILE_BYTES) + 1
    return CompilerAllocatedDFBConfig(
        dfb_index=0,
        num_tiles=overflow_tiles,
        data_format="bfloat16",
        block_count=1,
    )


class TestReducedWorkerL1:
    """Reduced worker_l1_size lowers cb_limit, triggering the budget check."""

    def test_overflow(self):
        default_size = ttnn.device.get_max_worker_l1_unreserved_size()
        reduced_size = default_size - 1_200_000
        device = ttnn.open_device(device_id=0, worker_l1_size=reduced_size)
        try:
            probe = to_l1(torch.zeros(32, 32, dtype=torch.bfloat16), device)
            remaining = get_min_remaining_l1_for_device(device)

            assert (
                remaining < DEFAULT_L1_CB_BUDGET_BYTES
            ), f"Expected reduced budget, got {remaining} >= {DEFAULT_L1_CB_BUDGET_BYTES}"

            with pytest.raises(ValueError, match="exceeds L1 budget"):
                build_cb_descriptors([probe], [_overflow_config(remaining)], None)
        finally:
            ttnn.close_device(device)


class TestL1TensorAllocation:
    """L1 tensor allocations reduce remaining budget on the target core."""

    def test_overflow(self):
        device = ttnn.open_device(device_id=0)
        try:
            remaining_empty = get_min_remaining_l1_for_device(device)

            # Height-sharded tensor on core (0,0) concentrates all L1 usage
            # on the target core.  512x512 bf16 = 256 tiles = 512 KiB.
            big = to_l1_sharded(torch.zeros(512, 512, dtype=torch.bfloat16), device)
            remaining_after = get_min_remaining_l1_for_device(device)

            assert remaining_after < remaining_empty, (
                f"Expected L1 tensor to reduce remaining: "
                f"{remaining_after} >= {remaining_empty}"
            )

            with pytest.raises(ValueError, match="exceeds L1 budget"):
                build_cb_descriptors([big], [_overflow_config(remaining_after)], None)
        finally:
            ttnn.close_device(device)


class TestBothReducedL1AndTensorAllocation:
    """Reduced worker_l1_size combined with L1 tensor allocations."""

    def test_overflow(self):
        default_size = ttnn.device.get_max_worker_l1_unreserved_size()
        reduced_size = default_size - 800_000
        device = ttnn.open_device(device_id=0, worker_l1_size=reduced_size)
        try:
            remaining_before_tensor = get_min_remaining_l1_for_device(device)

            big = to_l1_sharded(torch.zeros(256, 256, dtype=torch.bfloat16), device)
            remaining = get_min_remaining_l1_for_device(device)

            assert remaining < remaining_before_tensor, (
                f"Expected tensor to further reduce remaining: "
                f"{remaining} >= {remaining_before_tensor}"
            )
            assert remaining < DEFAULT_L1_CB_BUDGET_BYTES, (
                f"Expected combined budget < default: "
                f"{remaining} >= {DEFAULT_L1_CB_BUDGET_BYTES}"
            )

            with pytest.raises(ValueError, match="exceeds L1 budget"):
                build_cb_descriptors([big], [_overflow_config(remaining)], None)
        finally:
            ttnn.close_device(device)
