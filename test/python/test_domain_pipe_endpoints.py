# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-selected Pipe endpoints retain device and node coordinates."""

from ttl.domains import DeviceDomain
from ttl.pipe import Pipe, PipeNet


def test_two_galaxy_pipe_preserves_device_selected_endpoints():
    cluster = DeviceDomain((2, 8, 4))
    pipe = Pipe(
        src=cluster[0, 3, 2].at_node(11, 9),
        dst=cluster[1, 3, 2].at_node(0, 0),
    )
    net = PipeNet([pipe])
    assert net.is_graph
    relation_graph, relation_pipes = net._device_relations[0]
    edge = next(relation_graph.iter_edges())
    assert cluster.flattened_coordinates(edge.source) == (0, 3, 2)
    assert cluster.flattened_coordinates(edge.destination) == (1, 3, 2)
    assert relation_pipes[0].src == (11, 9)
    assert relation_pipes[0].dst == (0, 0)


def test_nested_view_and_direct_pipe_have_equal_compilation_identity():
    devices = DeviceDomain((8, 4))
    direct = Pipe(devices[3, 0].at_node(1, 0), devices[3, 2].at_node(0, 0))
    nested = Pipe(devices[3, :][0].at_node(1, 0), devices[3, :][2].at_node(0, 0))
    assert direct._operation_identity_capture() == nested._operation_identity_capture()
    assert (
        PipeNet([direct])._operation_identity_capture()
        == PipeNet([nested])._operation_identity_capture()
    )


def test_device_identity_distinguishes_otherwise_equal_node_pipes():
    devices = DeviceDomain((8, 4))
    first_row = Pipe(devices[0, 0].at_node(1, 0), devices[0, 1].at_node(0, 0))
    last_row = Pipe(devices[7, 0].at_node(1, 0), devices[7, 1].at_node(0, 0))
    assert (
        first_row._operation_identity_capture()
        != last_row._operation_identity_capture()
    )


def test_legacy_local_pipe_identity_is_preserved():
    local = Pipe(src=(1, 0), dst=(0, 0))
    assert local._operation_identity_capture() == (
        "pipe",
        (1, 0),
        (0, 0),
        (0, 0),
        False,
    )
    assert not PipeNet([local]).is_graph


def test_two_galaxy_pairing_has_32_edges_and_one_node_pipe():
    cluster = DeviceDomain((2, 8, 4))
    forward = Pipe.pairwise(
        src=cluster[0, :, :].at_node(11, 9),
        dst=cluster[1, :, :].at_node(0, 0),
    )
    relation_graph, relation_pipes = PipeNet([forward])._device_relations[0]
    edges = list(relation_graph.iter_edges())
    assert len(edges) == 32
    assert len(relation_pipes) == 1
    for edge in edges:
        source = cluster.flattened_coordinates(edge.source)
        assert cluster.flattened_coordinates(edge.destination) == (1, *source[1:])


def test_submesh_pairing_uses_view_coordinates():
    cluster = DeviceDomain((2, 8, 4))
    paired = Pipe.pairwise(
        src=cluster[0, 0:4, 0:2].at_node(11, 9),
        dst=cluster[1, 4:8, 2:4].at_node(0, 0),
    )
    relation_graph, _ = PipeNet([paired])._device_relations[0]
    edges = list(relation_graph.iter_edges())
    assert len(edges) == 8
    for edge in edges:
        source = cluster.flattened_coordinates(edge.source)
        assert cluster.flattened_coordinates(edge.destination) == (
            1,
            source[1] + 4,
            source[2] + 2,
        )


def test_same_device_pipe_has_one_local_relation():
    devices = DeviceDomain((8, 4))
    pipe = Pipe(devices[3, 2].at_node(1, 0), devices[3, 2].at_node(0, 0))

    relations = PipeNet([pipe])._device_relations

    assert len(relations) == 1
    relation_graph, relation_pipes = relations[0]
    edge = next(relation_graph.iter_edges())
    assert edge.source == edge.destination
    assert relation_pipes[0].src == (1, 0)
    assert relation_pipes[0].dst == (0, 0)


def test_row_all_to_all_separates_local_and_remote_transfers():
    devices = DeviceDomain((8, 4))
    pipe = Pipe.all_to_all(
        src=devices[3, :].at_node(0, 0),
        dst=devices[3, :].at_node(0, 0),
        include_self=True,
    )

    local_relation, remote_relation = PipeNet([pipe])._device_relations
    local_edges = tuple(local_relation[0].iter_edges())
    remote_edges = tuple(remote_relation[0].iter_edges())

    assert len(local_edges) == 4
    assert all(edge.source == edge.destination for edge in local_edges)
    assert len(remote_edges) == 12
    assert all(edge.source != edge.destination for edge in remote_edges)
    assert local_relation[1][0].src == remote_relation[1][0].src == (0, 0)
    assert local_relation[1][0].dst == remote_relation[1][0].dst == (0, 0)


def test_row_all_to_all_can_exclude_local_transfers():
    devices = DeviceDomain((8, 4))
    pipe = Pipe.all_to_all(
        src=devices[3, :].at_node(0, 0),
        dst=devices[3, :].at_node(0, 0),
    )

    relations = PipeNet([pipe])._device_relations

    assert len(relations) == 1
    assert len(tuple(relations[0][0].iter_edges())) == 12
