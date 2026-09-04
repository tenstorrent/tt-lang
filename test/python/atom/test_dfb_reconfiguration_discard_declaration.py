# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host tests for state-discarding DFB reconfiguration declarations."""

import pytest

import ttl


def _make_boundary(discard_dfb_state):
    return ttl.DFBReconfiguration(
        participants=(
            ttl.Kernel(ttl.KernelKind.COMPUTE),
            ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT),
            ttl.Kernel(ttl.KernelKind.DATA_MOVEMENT),
        ),
        discard_dfb_state=discard_dfb_state,
    )


def _make_operation(discard_dfb_state):
    boundary = _make_boundary(discard_dfb_state)

    @ttl.operation()
    def reconfiguration_operation():
        ttl.reconfigure_dfbs(boundary)

    return reconfiguration_operation


def test_discard_dfb_state_requires_bool():
    """The declaration rejects integer truth values."""
    with pytest.raises(TypeError, match="discard_dfb_state must be a bool"):
        _make_boundary(discard_dfb_state=1)


def test_discard_dfb_state_changes_operation_identity():
    """The cache identity distinguishes state-discarding boundaries."""
    preserving = _make_operation(discard_dfb_state=False)
    discarding = _make_operation(discard_dfb_state=True)

    assert preserving._spec.operation_identity != discarding._spec.operation_identity


def test_discard_dfb_state_survives_operation_composition():
    """Composition retains the reconfiguration's state-discarding contract."""
    nested_operation = _make_operation(discard_dfb_state=True)

    @ttl.operation()
    def composed_operation():
        nested_operation()

    boundaries = tuple(composed_operation._spec.dfb_reconfigurations.values())
    assert len(boundaries) == 1
    assert boundaries[0].discard_dfb_state
