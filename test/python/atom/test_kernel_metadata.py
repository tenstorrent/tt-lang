# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Off-device tests for logical-kernel metadata recovery from MLIR."""

import pytest

from ttl.dialects import ttl as ttl_dialect
from ttl.ir import Context, Module
from ttl.kernel import Kernel, KernelKind
from ttl.ttl_api import _get_kernel_logical_selector


_LOGICAL_KERNEL_MODULE = """
module {
  func.func @canonical_compute() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = compute>} { return }
  func.func @named_dm() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "mod.op">} { return }
  func.func @role_dm() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "<pipe_source>", role = "pipe_source">} { return }
  func.func @no_attribute() { return }
  func.func @wrong_type() attributes {ttl.logical_kernel = "not an attribute"} { return }
}
"""


@pytest.fixture
def logical_kernel_module():
    context = Context()
    ttl_dialect.ensure_dialects_registered(context)
    return Module.parse(_LOGICAL_KERNEL_MODULE, context)


def test_recovers_canonical_kernel_kind(logical_kernel_module):
    recovered = _get_kernel_logical_selector(logical_kernel_module, "canonical_compute")
    assert recovered is KernelKind.COMPUTE


def test_recovers_named_kernel_identity(logical_kernel_module):
    recovered = _get_kernel_logical_selector(logical_kernel_module, "named_dm")
    expected = Kernel._from_metadata(KernelKind.DATA_MOVEMENT, "reader", "mod.op")
    assert recovered == expected


def test_recovers_implicit_kernel_role(logical_kernel_module):
    recovered = _get_kernel_logical_selector(logical_kernel_module, "role_dm")
    expected = Kernel._from_metadata(
        KernelKind.DATA_MOVEMENT,
        "<pipe_source>",
        operation_identity=None,
        implicit_role="pipe_source",
    )
    assert recovered == expected


def test_missing_logical_kernel_metadata_returns_none(logical_kernel_module):
    assert _get_kernel_logical_selector(logical_kernel_module, "no_attribute") is None


def test_wrong_logical_kernel_attribute_type_is_rejected(logical_kernel_module):
    with pytest.raises(ValueError, match="Invalid 'ttl.logical_kernel'"):
        _get_kernel_logical_selector(logical_kernel_module, "wrong_type")
