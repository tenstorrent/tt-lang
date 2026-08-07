# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator behavior for tensor-backed dataflow buffers."""

import pytest

from sim import ttl


def test_tensor_backed_dfb_is_explicitly_unsupported():
    with pytest.raises(
        NotImplementedError,
        match="simulator does not model tensor-backed DFB storage",
    ):
        ttl.make_tensor_backed_dfb(object(), shape=(1, 1))


def test_tensor_backed_dfb_rejection_preserves_simulator_namespace():
    assert ttl.TILE_LAYOUT is not None
    assert ttl.ROW_MAJOR_LAYOUT is not None
    assert ttl.Program is not None
    assert ttl.block is not None
    assert ttl.math is not None
