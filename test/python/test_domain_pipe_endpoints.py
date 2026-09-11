# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Complete Pipe endpoints normalize to parent device and node coordinates."""

from ttl.domains import DeviceDomain
from ttl.pipe import Pipe, PipeNet


def test_two_galaxy_pipe_preserves_complete_endpoints():
    cluster = DeviceDomain((2, 8, 4))
    pipe = Pipe(
        src=cluster[0, 3, 2].at_node(11, 9),
        dst=cluster[1, 3, 2].at_node(0, 0),
    )
    net = PipeNet([pipe])
    assert net.is_graph
    mapping = net.mappings[0]
    edge = next(mapping.graph.iter_edges())
    assert cluster.flattened_coordinates(edge.source) == (0, 3, 2)
    assert cluster.flattened_coordinates(edge.destination) == (1, 3, 2)
    assert mapping.pipes[0].src == (11, 9)
    assert mapping.pipes[0].dst == (0, 0)


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
    mapping = PipeNet([forward]).mappings[0]
    edges = list(mapping.graph.iter_edges())
    assert len(edges) == 32
    assert len(mapping.pipes) == 1
    for edge in edges:
        source = cluster.flattened_coordinates(edge.source)
        assert cluster.flattened_coordinates(edge.destination) == (1, *source[1:])


def test_submesh_pairing_uses_view_coordinates():
    cluster = DeviceDomain((2, 8, 4))
    paired = Pipe.pairwise(
        src=cluster[0, 0:4, 0:2].at_node(11, 9),
        dst=cluster[1, 4:8, 2:4].at_node(0, 0),
    )
    edges = list(PipeNet([paired]).mappings[0].graph.iter_edges())
    assert len(edges) == 8
    for edge in edges:
        source = cluster.flattened_coordinates(edge.source)
        assert cluster.flattened_coordinates(edge.destination) == (
            1,
            source[1] + 4,
            source[2] + 2,
        )
