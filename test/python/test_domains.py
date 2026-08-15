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
    StencilTransfer,
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


def test_device_ref_identity_ignores_construction_names():
    named = DeviceRef(board=0, device=1)
    positional = DeviceRef((0,), (1,))

    assert named == positional
    assert hash(named) == hash(positional)


def test_product_domain_rejects_missing_component():
    domain = DeviceDomain.product(board=(2,), device=(4,))

    with pytest.raises(ValueError, match="do not match domain components"):
        domain.resolve_device_ref(DeviceRef(board=0))


def test_transfer_graph_does_not_apply_target_routability_rules():
    domain = DeviceDomain((2, 4))

    graph = TransferGraph.edges(domain, edges=[((0, 0), (1, 0))])

    assert graph.transfer_edges[0].destination.coordinates == ((1, 0),)


def test_direct_transfer_graph_construction_validates_edges():
    domain = DeviceDomain((4,))

    with pytest.raises(ValueError, match="0 <= coord < 4"):
        TransferGraph(
            domain,
            edges=[TransferEdge(DeviceRef((0,)), DeviceRef((4,)))],
        )


def test_transfer_graph_rejects_exact_self_transfer():
    domain = DeviceDomain((4,))

    with pytest.raises(ValueError, match="source must differ from destination"):
        TransferGraph.edges(domain, edges=[(1, 1)])


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


# An extent-two wrapped relation has no distinct boundary edge; it is the
# bidirectional ordinary-neighbor pair.
def test_axis_neighbor_extent_two_reuses_ordinary_neighbor():
    domain = DeviceDomain((2,))

    graph = TransferGraph.axis_neighbor(domain, wrap=True)

    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0,)), DeviceRef((1,))),
        TransferEdge(DeviceRef((1,)), DeviceRef((0,))),
    ]


def test_stencil_edges_include_multiple_axes_and_directions():
    domain = DeviceDomain((2, 2))

    graph = TransferGraph.stencil(
        domain,
        offsets=[(-1, 0), (1, 0), (0, -1), (0, 1)],
    )

    assert isinstance(graph.structured, StencilTransfer)
    assert set(graph.iter_edges()) == {
        TransferEdge(DeviceRef((0, 0)), DeviceRef((1, 0))),
        TransferEdge(DeviceRef((0, 0)), DeviceRef((0, 1))),
        TransferEdge(DeviceRef((0, 1)), DeviceRef((1, 1))),
        TransferEdge(DeviceRef((0, 1)), DeviceRef((0, 0))),
        TransferEdge(DeviceRef((1, 0)), DeviceRef((0, 0))),
        TransferEdge(DeviceRef((1, 0)), DeviceRef((1, 1))),
        TransferEdge(DeviceRef((1, 1)), DeviceRef((0, 1))),
        TransferEdge(DeviceRef((1, 1)), DeviceRef((1, 0))),
    }


def test_stencil_wrap_deduplicates_equivalent_edges():
    domain = DeviceDomain((2,))

    graph = TransferGraph.stencil(
        domain,
        offsets=[(-1,), (1,)],
        wrap=True,
    )

    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0,)), DeviceRef((1,))),
        TransferEdge(DeviceRef((1,)), DeviceRef((0,))),
    ]


def test_stencil_preserves_unselected_product_components():
    domain = DeviceDomain.product(board=(2,), device=(2, 2))

    graph = TransferGraph.stencil(
        domain,
        component="device",
        offsets=[(0, 1)],
    )

    assert list(graph.iter_edges()) == [
        TransferEdge(
            DeviceRef((0,), (0, 0)),
            DeviceRef((0,), (0, 1)),
        ),
        TransferEdge(
            DeviceRef((0,), (1, 0)),
            DeviceRef((0,), (1, 1)),
        ),
        TransferEdge(
            DeviceRef((1,), (0, 0)),
            DeviceRef((1,), (0, 1)),
        ),
        TransferEdge(
            DeviceRef((1,), (1, 0)),
            DeviceRef((1,), (1, 1)),
        ),
    ]


@pytest.mark.parametrize(
    "offsets,message",
    [
        ([], "at least one offset"),
        ([(0, 0)], "zero offset"),
        ([(1,)], "has rank 1, expected 2"),
        ([(1, 0), (1, 0)], "must be unique"),
    ],
)
def test_stencil_rejects_invalid_offsets(offsets, message):
    domain = DeviceDomain((2, 2))

    with pytest.raises(ValueError, match=message):
        TransferGraph.stencil(domain, offsets=offsets)


def test_gather_edges_preserve_other_product_components():
    domain = DeviceDomain.product(board=(2,), device=(2,))

    graph = TransferGraph.gather(domain, DeviceRef(device=0), component="device")

    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0,), (1,)), DeviceRef((0,), (0,))),
        TransferEdge(DeviceRef((1,), (1,)), DeviceRef((1,), (0,))),
    ]


def test_scatter_edges_preserve_other_product_components():
    domain = DeviceDomain.product(board=(2,), device=(2,))

    graph = TransferGraph.scatter(domain, DeviceRef(device=0), component="device")

    assert list(graph.iter_edges()) == [
        TransferEdge(DeviceRef((0,), (0,)), DeviceRef((0,), (1,))),
        TransferEdge(DeviceRef((1,), (0,)), DeviceRef((1,), (1,))),
    ]


def test_component_transfer_rejects_full_product_endpoint():
    domain = DeviceDomain.product(board=(2,), device=(2,))

    with pytest.raises(ValueError, match="must name only component 'device'"):
        TransferGraph.scatter(
            domain,
            DeviceRef(board=0, device=0),
            component="device",
        )


@pytest.mark.parametrize(
    "create_graph",
    [
        lambda: TransferGraph.axis_neighbor(DeviceDomain((1, 3)), axis=1, offset=3),
        lambda: TransferGraph.axis_neighbor(
            DeviceDomain((1, 3)), axis=1, offset=3, wrap=True
        ),
        lambda: TransferGraph.stencil(DeviceDomain((1, 3)), offsets=[(0, 3)]),
        lambda: TransferGraph.stencil(
            DeviceDomain((1, 3)), offsets=[(0, 3)], wrap=True
        ),
        lambda: TransferGraph.gather(DeviceDomain((1,)), 0),
        lambda: TransferGraph.scatter(DeviceDomain((1,)), 0),
        lambda: TransferGraph.all_to_all(DeviceDomain((1,))),
    ],
    ids=[
        "axis-neighbor-out-of-bounds",
        "axis-neighbor-wrap-self-transfer",
        "stencil-out-of-bounds",
        "stencil-wrap-self-transfer",
        "gather-single-device",
        "scatter-single-device",
        "all-to-all-single-device",
    ],
)
def test_structured_graph_rejects_empty_relation(create_graph):
    with pytest.raises(ValueError, match="relation contains no edges"):
        create_graph()


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


def test_device_domain_index_order_is_row_major():
    rectangular_domain = DeviceDomain((2, 3))
    product_domain = DeviceDomain.product(board=(2,), device=(3,))

    assert rectangular_domain.index_order((0, 0)) == 0
    assert rectangular_domain.index_order((1, 2)) == 5
    assert product_domain.index_order(DeviceRef(board=0, device=0)) == 0
    assert product_domain.index_order(DeviceRef(board=1, device=2)) == 5


def test_device_domain_current_index_is_kernel_only():
    domain = DeviceDomain((2, 3))

    with pytest.raises(RuntimeError, match="only be called inside a TTL kernel"):
        domain.current_index()


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

    with pytest.raises(ValueError, match="source must not be contained"):
        TransferGraph.edges(
            domain,
            edges=[
                (
                    (0, 1),
                    DeviceRange(lo=DeviceRef((0, 0)), hi=DeviceRef((1, 3))),
                )
            ],
        )
