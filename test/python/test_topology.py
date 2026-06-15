# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for logical device domains and transfer graphs."""

import pytest

from ttl.topology import (
    DeviceDomain,
    DeviceRange,
    DeviceRef,
    Fabric1D,
    FabricRing,
    FabricTopology,
    ProjectedTransferFamily,
    TopologyLevelInfo,
    TransferGraph,
)


def test_explicit_flat_edge_projects_to_fabric_axis():
    domain = DeviceDomain((1, 4), topology=Fabric1D(axis=1))
    graph = TransferGraph.edges(domain, edges=[((0, 0), (0, 2))])

    projections = graph.project_initial()

    assert graph.explicit_edge_count == 1
    assert len(projections) == 1
    projection = projections[0]
    assert not projection.is_local
    assert projection.level_name == "device"
    assert projection.source_level_coordinate == (0, 0)
    assert projection.destination_level_coordinate == (0, 2)


def test_topology_rejects_periodic_mismatch():
    with pytest.raises(ValueError, match="fabric_ring topology requires periodic=True"):
        FabricTopology(kind="fabric_ring", cluster_axis=0, periodic=False)

    with pytest.raises(ValueError, match="fabric_1d topology requires periodic=False"):
        FabricTopology(kind="fabric_1d", cluster_axis=0, periodic=True)


def test_coordinate_validation_uses_domain_extent():
    domain = DeviceDomain((1, 4), topology=Fabric1D(axis=1))

    with pytest.raises(ValueError, match="0 <= coord < 4"):
        TransferGraph.edges(domain, edges=[((0, 0), (0, 4))])


def test_hierarchical_edge_projects_when_one_level_differs():
    domain = DeviceDomain.hierarchy(
        TopologyLevelInfo("board", extent=(2,), topology=FabricRing(axis=0)),
        TopologyLevelInfo("device", extent=(4,), topology=Fabric1D(axis=0)),
    )
    graph = TransferGraph.edges(
        domain,
        edges=[
            (
                DeviceRef(board=0, device=1),
                DeviceRef(board=1, device=1),
            )
        ],
    )

    projections = graph.project_initial()

    assert len(projections) == 1
    assert projections[0].level_name == "board"
    assert projections[0].source_level_coordinate == (0,)
    assert projections[0].destination_level_coordinate == (1,)


def test_hierarchical_edge_rejects_multi_level_route():
    domain = DeviceDomain.hierarchy(
        TopologyLevelInfo("board", extent=(2,), topology=FabricRing(axis=0)),
        TopologyLevelInfo("device", extent=(4,), topology=Fabric1D(axis=0)),
    )
    graph = TransferGraph.edges(
        domain,
        edges=[
            (
                DeviceRef(board=0, device=1),
                DeviceRef(board=1, device=2),
            )
        ],
    )

    with pytest.raises(ValueError, match="MD-14"):
        graph.project_initial()


def test_initial_projection_rejects_cross_axis_remote_edge():
    domain = DeviceDomain((2, 4), topology=Fabric1D(axis=1))
    graph = TransferGraph.edges(domain, edges=[((0, 0), (1, 0))])

    with pytest.raises(ValueError, match="non-cluster axis 0"):
        graph.project_initial()


def test_structured_axis_neighbor_remains_compact():
    domain = DeviceDomain((1024, 1024), topology=Fabric1D(axis=1))
    graph = TransferGraph.axis_neighbor(domain, axis=1, offset=1)

    projection = graph.project_initial()

    assert graph.is_structured
    assert graph.explicit_edge_count is None
    assert graph.transfer_edges == ()
    assert isinstance(projection, ProjectedTransferFamily)
    assert projection.level_name == "device"
    assert "structured descriptor" in graph.metadata_cost().compile_time


def test_structured_hierarchical_gather_rejects_multi_level_route():
    domain = DeviceDomain.hierarchy(
        TopologyLevelInfo("board", extent=(2,), topology=FabricRing(axis=0)),
        TopologyLevelInfo("device", extent=(4,), topology=Fabric1D(axis=0)),
    )
    graph = TransferGraph.gather(domain, DeviceRef(board=0, device=0), level="board")

    with pytest.raises(ValueError, match="MD-14"):
        graph.project_initial()


def test_range_projection_rejects_source_in_destination():
    domain = DeviceDomain((1, 4), topology=Fabric1D(axis=1))
    graph = TransferGraph.edges(
        domain,
        edges=[
            (
                (0, 1),
                DeviceRange(lo=DeviceRef((0, 0)), hi=DeviceRef((1, 3))),
            )
        ],
    )

    with pytest.raises(ValueError, match="MD-6"):
        graph.project_initial()
