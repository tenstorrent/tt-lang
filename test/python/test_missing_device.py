# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for device error handling.

Verifies that meaningful error messages are produced when operations
receive host tensors instead of device tensors, and that tensors on
different devices are rejected.
"""

from unittest.mock import patch

import pytest
import torch
import ttl
from ttl.ttl_api import _require_device, _same_device

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)


@ttl.operation(grid="full")
def nop_full_grid(a):
    @ttl.compute()
    def compute_nop():
        pass

    @ttl.datamovement()
    def dm_nop1():
        pass

    @ttl.datamovement()
    def dm_nop2():
        pass


@ttl.operation(grid="full")
def nop_full_grid_2(a, b):
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


def test_full_grid_host_tensor():
    """
    grid='full' with a host tensor should produce a clear error, not an AttributeError on NoneType.
    """
    a_host = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    with pytest.raises(ValueError, match="No device found"):
        nop_full_grid(a_host)


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


def test_full_grid_no_ttnn_tensors():
    """
    grid='full' with no ttnn tensors should report that none were provided.
    The fixed-grid path hits _require_device at __call__ time (post-compile),
    so it can't be reached without a valid ttnn tensor to compile against.
    """
    with pytest.raises(ValueError, match="no ttnn tensor arguments were provided"):
        nop_full_grid(torch.zeros(32, 32, dtype=torch.bfloat16))


def test_full_grid_multiple_host_tensors():
    """Error message should list all host tensor arguments."""
    a_host = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    b_host = ttnn.from_torch(
        torch.zeros(64, 64, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    with pytest.raises(
        ValueError, match="All ttnn tensor inputs are on host"
    ) as exc_info:
        nop_full_grid_2(a_host, b_host)
    msg = str(exc_info.value)
    assert "arg[0]" in msg
    assert "arg[1]" in msg


def test_full_grid_mixed_host_and_device(device):
    """_require_device succeeds when at least one tensor is on-device."""

    a_host = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    b_device = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    assert _require_device((a_host, b_device)) is not None


# =========================================================================
# _same_device unit tests (no hardware required)
# =========================================================================


class _FakeDevice:
    """Minimal stand-in for a TTNN device with an integer id."""

    def __init__(self, dev_id):
        self._id = dev_id

    def id(self):
        return self._id

    def __repr__(self):
        return f"FakeDevice({self._id})"


def test_same_device_identity():

    d = _FakeDevice(0)
    assert _same_device(d, d) is True


def test_same_device_equal_ids():

    assert _same_device(_FakeDevice(5), _FakeDevice(5)) is True


def test_same_device_different_ids():

    assert _same_device(_FakeDevice(0), _FakeDevice(1)) is False


def test_same_device_no_id_method():
    """Objects without .id() are only equal by identity."""

    a, b = object(), object()
    assert _same_device(a, a) is True
    assert _same_device(a, b) is False


# =========================================================================
# _require_device same-device validation (mock, no hardware required)
# =========================================================================


class _FakeTensor:
    """Minimal stand-in for a TTNN tensor."""

    def __init__(self, dev):
        self._dev = dev
        self.shape = (32, 32)

    def device(self):
        return self._dev


def test_require_device_different_devices_raises():
    """Two tensors on different devices must raise ValueError."""

    dev0 = _FakeDevice(0)
    dev1 = _FakeDevice(1)

    with patch("ttl.ttl_api.is_ttnn_tensor", return_value=True):
        with pytest.raises(ValueError, match="different devices"):
            _require_device((_FakeTensor(dev0), _FakeTensor(dev1)))


def test_require_device_same_device_ok():
    """Two tensors on the same device should return that device."""

    dev = _FakeDevice(0)

    with patch("ttl.ttl_api.is_ttnn_tensor", return_value=True):
        result = _require_device((_FakeTensor(dev), _FakeTensor(dev)))
    assert result is dev


def test_require_device_same_id_different_objects():
    """Distinct device objects with equal .id() should be accepted."""

    with patch("ttl.ttl_api.is_ttnn_tensor", return_value=True):
        result = _require_device(
            (_FakeTensor(_FakeDevice(3)), _FakeTensor(_FakeDevice(3)))
        )
    assert result.id() == 3


def test_require_device_three_tensors_mismatch_at_third():
    """Mismatch detected on arg[2] when arg[0] and arg[1] agree."""

    dev_a = _FakeDevice(0)
    dev_b = _FakeDevice(1)

    with patch("ttl.ttl_api.is_ttnn_tensor", return_value=True):
        with pytest.raises(ValueError, match=r"arg\[2\]"):
            _require_device(
                (_FakeTensor(dev_a), _FakeTensor(dev_a), _FakeTensor(dev_b))
            )


# =========================================================================
# Real TTNN tensors on a single device (requires hardware)
# =========================================================================


def test_require_device_two_tensors_same_device(device):
    """Two real TTNN tensors on the same device should pass."""

    a = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    b = ttnn.from_torch(
        torch.zeros(32, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    result = _require_device((a, b))
    assert result is not None
