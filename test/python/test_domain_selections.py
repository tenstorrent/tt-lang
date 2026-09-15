# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parent identity and compact indexing for device and node selections."""

import pytest

import ttl.domains as domains
from ttl.domains import DeviceDomain, DevicePoint, DeviceView


def coordinates(selection):
    return [
        selection.domain.flattened_coordinates(reference)
        for reference in selection.iter_device_refs()
    ]


def test_galaxy_row_preserves_parent_coordinates():
    devices = DeviceDomain((8, 4))
    row = devices[5, :]
    assert row.shape == (4,)
    assert coordinates(row) == [(5, column) for column in range(4)]
    assert row[2] == devices[5, 2]
    assert row.at_node(1, 0).devices == row


def test_two_galaxy_submeshes_keep_distinct_identities():
    cluster = DeviceDomain((2, 8, 4))
    first = cluster[0, :, :]
    second = cluster[1, :, :]
    assert first.shape == second.shape == (8, 4)
    assert second[3, 2] == cluster[1, 3, 2]
    assert first[3, 2] != second[3, 2]
    assert len(coordinates(first)) == len(coordinates(second)) == 32


def test_product_components_survive_flattened_indexing():
    cluster = DeviceDomain.product(galaxy=(2,), mesh=(8, 4))
    point = cluster[1, 3, 2]
    assert point.reference.coordinates == ((1,), (3, 2))
    assert cluster.flattened_coordinates(point.reference) == (1, 3, 2)


def test_nested_strided_and_reverse_views():
    devices = DeviceDomain((8, 4))
    selected = devices[1:8:2, ::-1]
    assert coordinates(selected[1:, 1::2]) == [
        (row, column) for row in (3, 5, 7) for column in (2, 0)
    ]
    assert selected[-1, -1] == devices[7, 0]


def test_empty_view_keeps_rank_and_has_no_members():
    selected = DeviceDomain((8, 4))[2:2, :]
    assert selected.shape == (0, 4)
    assert coordinates(selected) == []
    with pytest.raises(IndexError):
        selected[0, 0]


def test_indexing_large_domain_does_not_expand_devices():
    selected = DeviceDomain((1000000, 1000000))[::2, 3]
    assert selected.axes == (range(0, 1000000, 2), 3)
    assert selected.shape == (500000,)


def test_sparse_selection_and_union_deduplicate_parent_members():
    devices = DeviceDomain((8, 4))
    sparse = devices.select([devices[7, 3], devices[0, 0], devices[7, 3]])
    assert coordinates(sparse) == [(0, 0), (7, 3)]
    assert coordinates(sparse | devices[0, :]) == [
        (0, column) for column in range(4)
    ] + [(7, 3)]


def test_selection_identity_includes_node_and_parent_coordinates():
    cluster = DeviceDomain((2, 8, 4))
    direct = cluster[1, 3, 2].at_node(1, 0)
    nested = cluster[1, :, :][3, 2].at_node(1, 0)
    assert direct._operation_identity_capture() == nested._operation_identity_capture()
    assert (
        direct._operation_identity_capture()
        != cluster[0, 3, 2].at_node(1, 0)._operation_identity_capture()
    )
    assert (
        direct._operation_identity_capture()
        != cluster[1, 3, 2].at_node(0, 0)._operation_identity_capture()
    )


def test_partial_indexing_and_direct_domain_node_selection():
    devices = DeviceDomain((8, 4))
    assert devices[2] == devices[2, :]
    assert devices.at_node(1, 0) == devices[:, :].at_node(1, 0)
    assert isinstance(devices[2, 3], DevicePoint)
    assert isinstance(devices[2], DeviceView)


def test_selection_types_are_exported_from_domains():
    assert {
        "DevicePoint",
        "DeviceSelection",
        "DeviceSet",
        "DeviceView",
        "NodeSelection",
    }.issubset(domains.__all__)
