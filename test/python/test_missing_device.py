# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for missing device error handling (GitHub issue #389).

Verifies that meaningful error messages are produced when operations
receive host tensors (after ttnn.from_device()) instead of device tensors.
"""

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


def test_auto_grid_host_tensor():
    """
    grid='auto' with a host tensor should produce a clear error, not an AttributeError on NoneType.
    """
    a_host = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    with pytest.raises(ValueError, match="No device found"):
        nop_auto_grid(a_host)


def test_fixed_grid_host_tensor():
    """
    grid=(1,1) with a host tensor should produce a clear error, not an AttributeError on NoneType.
    """
    a_host = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    with pytest.raises(ValueError, match="No device found"):
        nop_fixed_grid(a_host)


def test_auto_grid_no_ttnn_tensors():
    """
    grid='auto' with no ttnn tensors should report that none were provided.
    The fixed-grid path hits _require_device at __call__ time (post-compile),
    so it can't be reached without a valid ttnn tensor to compile against.
    """
    with pytest.raises(ValueError, match="no ttnn tensor arguments were provided"):
        nop_auto_grid(torch.zeros(32, 32, dtype=torch.bfloat16))
