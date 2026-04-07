# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for missing device error handling (GitHub issue #389).

Verifies that meaningful error messages are produced when operations
receive host tensors (after ttnn.from_device()) instead of device tensors.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch
import ttl

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)


@ttl.operation(grid="auto")
def nop_auto_grid(a):
    @ttl.compute()
    def compute_nop():
        pass

    @ttl.datamovement()
    def dm_nop1():
        pass

    @ttl.datamovement()
    def dm_nop2():
        pass


@ttl.operation(grid=(1, 1))
def nop_fixed_grid(a):
    @ttl.compute()
    def compute_nop():
        pass

    @ttl.datamovement()
    def dm_nop1():
        pass

    @ttl.datamovement()
    def dm_nop2():
        pass


def test_auto_grid_host_tensor(device):
    """grid='auto' with a host tensor should produce a clear error, not an AttributeError on NoneType."""
    a = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    a_host = ttnn.from_device(a)

    with pytest.raises(ValueError, match="No device found"):
        nop_auto_grid(a_host)


def test_fixed_grid_host_tensor(device):
    """grid=(1,1) with a host tensor should produce a clear error, not an AttributeError on NoneType."""
    a = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    a_host = ttnn.from_device(a)

    with pytest.raises(ValueError, match="No device found"):
        nop_fixed_grid(a_host)
