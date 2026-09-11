# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reject ambiguous or incompatible complete Pipe endpoints."""

import pytest

from ttl.domains import DeviceDomain
from ttl.pipe import Pipe, PipeNet


def test_mixed_local_and_complete_endpoint_rejected():
    with pytest.raises(TypeError, match="both Pipe endpoints"):
        Pipe(DeviceDomain((8, 4))[0, 0].at_node(1, 0), (0, 0))


def test_incompatible_domains_rejected():
    with pytest.raises(ValueError, match="parent device domain"):
        Pipe(
            DeviceDomain((8, 4))[0, 0].at_node(1, 0),
            DeviceDomain((2, 8, 4))[1, 0, 0].at_node(0, 0),
        )


def test_ambiguous_device_selections_rejected():
    devices = DeviceDomain((8, 4))
    with pytest.raises(ValueError, match="relation constructor"):
        Pipe(devices[0, :].at_node(1, 0), devices[1, :].at_node(0, 0))


def test_mixed_relative_and_complete_pipes_rejected():
    devices = DeviceDomain((8, 4))
    complete = Pipe(devices[0, 0].at_node(1, 0), devices[0, 1].at_node(0, 0))
    with pytest.raises(ValueError, match="mix complete endpoints"):
        PipeNet([complete, Pipe((1, 0), (0, 0))])


def test_pairwise_equal_counts_do_not_imply_equal_extents():
    devices = DeviceDomain((8, 4))
    with pytest.raises(ValueError, match="equal extents"):
        Pipe.pairwise(
            src=devices[:4, :2].at_node(1, 0), dst=devices[4:6, :].at_node(0, 0)
        )


def test_pairwise_sparse_sets_require_explicit_correspondence():
    devices = DeviceDomain((8, 4))
    sparse = devices.select([devices[0, 0], devices[7, 3]])
    with pytest.raises(TypeError, match="coordinate-structured"):
        Pipe.pairwise(src=sparse.at_node(1, 0), dst=devices[1, :2].at_node(0, 0))
