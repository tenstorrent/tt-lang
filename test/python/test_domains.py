# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-only tests for logical device domains and transfer graphs."""

import pytest

from ttl.domains import (
    AllToAllTransfer,
    AxisNeighborTransfer,
    DeviceDomain,
    DeviceRange,
    DeviceRef,
    TransferEdge,
    TransferGraph,
)


def test_explicit_edge_uses_regular_domain_coordinates():
    domain = DeviceDomain((1, 4))
    graph = TransferGraph.edges(domain, edges=[((0, 0), (0, 2))])

    assert graph.explicit_edge_count == 1
    assert graph.transfer_edges[0].source.coordinates == ((0, 0),)
    assert graph.transfer_edges[0].destination.coordinates == ((0, 2),)


def test_domain_rejects_architecture_topology_arguments():
    with pytest.raises(TypeError, match="unexpected keyword argument 'topology'"):
        DeviceDomain((1, 4), topology="fabric_1d")


def test_coordinate_validation_uses_domain_extent():
    domain = DeviceDomain((1, 4))

    with pytest.raises(ValueError, match="0 <= coord < 4"):
        TransferGraph.edges(domain, edges=[((0, 0), (0, 4))])


def test_product_domain_resolves_named_device_refs():
    domain = DeviceDomain.product(
        board=DeviceDomain((2,)),
        device=DeviceDomain((4,)),
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

    edge = graph.transfer_edges[0]
    assert edge.source.coordinates == ((0,), (1,))
    assert edge.destination.coordinates == ((1,), (2,))


def test_product_domain_rejects_missing_component():
    domain = DeviceDomain.product(board=(2,), device=(4,))

    with pytest.raises(ValueError, match="do not match domain components"):
        domain.resolve_device_ref(DeviceRef(board=0))


def test_transfer_graph_does_not_apply_target_routability_rules():
    domain = DeviceDomain((2, 4))

    graph = TransferGraph.edges(domain, edges=[((0, 0), (1, 0))])

    assert graph.transfer_edges[0].destination.coordinates == ((1, 0),)


def test_structured_axis_neighbor_remains_compact():
    domain = DeviceDomain((1024, 1024))
    graph = TransferGraph.axis_neighbor(domain, axis=1, offset=1)

    assert graph.is_structured
    assert graph.explicit_edge_count is None
    assert graph.transfer_edges == ()
    assert graph.structured.component_name == "device"
    assert graph.structured.axis == 1
    assert "structured descriptor" in graph.metadata_cost().compile_time


def test_axis_neighbor_edges_are_materialized_from_compact_relation():
    domain = DeviceDomain((1, 3))

    graph = TransferGraph.axis_neighbor(domain, axis=1, wrap=True)

    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0, 0)), DeviceRef((0, 1))),
        TransferEdge(DeviceRef((0, 1)), DeviceRef((0, 2))),
        TransferEdge(DeviceRef((0, 2)), DeviceRef((0, 0))),
    ]


def test_gather_edges_preserve_other_product_components():
    domain = DeviceDomain.product(board=(2,), device=(2,))

    graph = TransferGraph.gather(
        domain, DeviceRef(board=0, device=0), component="device"
    )

    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0,), (1,)), DeviceRef((0,), (0,))),
        TransferEdge(DeviceRef((1,), (1,)), DeviceRef((1,), (0,))),
    ]


def test_multicast_edges_preserve_other_product_components():
    domain = DeviceDomain.product(board=(2,), device=(2,))

    graph = TransferGraph.multicast(
        domain, DeviceRef(board=0, device=0), component="device"
    )

    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0,), (0,)), DeviceRef((0,), (1,))),
        TransferEdge(DeviceRef((1,), (0,)), DeviceRef((1,), (1,))),
    ]


def test_all_to_all_edges_preserve_other_product_components():
    domain = DeviceDomain.product(board=(2,), device=(2,))

    graph = TransferGraph.all_to_all(domain, component="device")

    assert isinstance(graph.structured, AllToAllTransfer)
    assert graph.transfer_edges == ()
    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0,), (0,)), DeviceRef((0,), (1,))),
        TransferEdge(DeviceRef((0,), (1,)), DeviceRef((0,), (0,))),
        TransferEdge(DeviceRef((1,), (0,)), DeviceRef((1,), (1,))),
        TransferEdge(DeviceRef((1,), (1,)), DeviceRef((1,), (0,))),
    ]


def test_device_domain_current_predicate_is_kernel_only():
    domain = DeviceDomain((1, 4))

    with pytest.raises(RuntimeError, match="only be called inside a TTL kernel"):
        domain.is_current((0, 0))

    with pytest.raises(ValueError, match="0 <= coord < 4"):
        domain.is_current((0, 4))


def test_product_structured_transfer_requires_component():
    domain = DeviceDomain.product(board=(2,), device=(4,))

    with pytest.raises(ValueError, match="require an explicit component"):
        TransferGraph.axis_neighbor(domain)

    graph = TransferGraph.axis_neighbor(domain, component="device")
    assert graph.structured.component_name == "device"


def test_direct_axis_neighbor_transfer_construction_is_validated():
    domain = DeviceDomain((4,))

    with pytest.raises(ValueError, match="offset must be positive"):
        TransferGraph(
            domain,
            structured=AxisNeighborTransfer(
                component_name="device",
                axis=0,
                offset=0,
                wrap=False,
            ),
        )


def test_range_rejects_source_in_destination():
    domain = DeviceDomain((1, 4))

    with pytest.raises(ValueError, match="source-in-destination"):
        TransferGraph.edges(
            domain,
            edges=[
                (
                    (0, 1),
                    DeviceRange(lo=DeviceRef((0, 0)), hi=DeviceRef((1, 3))),
                )
            ],
        )
