# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Negative tests for ttl.PipeNet construction-time validation.

Pure Python tests: no device, no MLIR. Pin the user-visible error
contract for invalid PipeNet shapes.
"""

import pytest
import ttl
import ttl.ttl_api as ttl_api

from ttl._pipenets import iter_referenced_function_values


def test_empty_pipenet_rejected():
    with pytest.raises(ValueError, match="at least one pipe"):
        ttl.PipeNet([])


def test_pipenet_requires_exactly_one_representation():
    domain = ttl.DeviceDomain((1, 2))
    graph = ttl.TransferGraph.edges(domain, edges=[((0, 0), (0, 1))])

    with pytest.raises(ValueError, match="exactly one of pipes or graph"):
        ttl.PipeNet()
    with pytest.raises(ValueError, match="exactly one of pipes or graph"):
        ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))], graph=graph)


def test_pipenet_accepts_transfer_graph():
    domain = ttl.DeviceDomain((1, 2))
    graph = ttl.TransferGraph.edges(domain, edges=[((0, 0), (0, 1))])

    net = ttl.PipeNet(graph=graph)

    assert net.is_graph
    assert net.graph is graph
    assert net.pipes == []


def test_pipenet_graph_requires_transfer_graph():
    with pytest.raises(TypeError, match="TransferGraph"):
        ttl.PipeNet(graph=object())


def test_pipenet_graph_defers_target_routability_validation():
    domain = ttl.DeviceDomain((2, 2))
    graph = ttl.TransferGraph.edges(domain, edges=[((0, 0), (1, 0))])

    net = ttl.PipeNet(graph=graph)

    assert net.graph is graph


def test_graph_pipenet_requires_matching_operation_device_domain():
    graph_domain = ttl.DeviceDomain((1, 2))
    net = ttl.PipeNet(
        graph=ttl.TransferGraph.edges(graph_domain, edges=[((0, 0), (0, 1))])
    )

    def operation():
        return net

    with pytest.raises(ValueError, match="requires operation device_domain"):
        ttl_api._build_operation_pipenets(operation, [])
    with pytest.raises(ValueError, match="must match operation device_domain"):
        ttl_api._build_operation_pipenets(operation, [], ttl.DeviceDomain((1, 3)))

    pipenets = ttl_api._build_operation_pipenets(
        operation, [], ttl.DeviceDomain((1, 2))
    )
    assert len(pipenets.graph_pipe_nets) == 1


def test_capture_discovery_ignores_empty_closure_cells():
    def make_function_with_empty_cell():
        value = object()

        def operation():
            return value

        del value
        return operation

    operation = make_function_with_empty_cell()

    assert list(iter_referenced_function_values(operation)) == []


def test_within_pipenet_overlapping_collective_dst_allowed():
    """Two collective pipes whose destination rectangles intersect inside a
    single PipeNet are allowed.

    The two pipes both target column 1 rows 0..3, so the node at (1, 1)
    receives from both. Per-PipeNet receiver counters disambiguate the
    handshake.
    """
    ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(1, slice(0, 4))),
            ttl.Pipe(src=(2, 0), dst=(1, slice(0, 4))),
        ]
    )


def test_within_pipenet_partially_overlapping_collective_dst_allowed():
    """Collective destinations that overlap on even one node are allowed."""
    ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(slice(0, 3), 0)),  # nodes 0..2 row 0
            ttl.Pipe(src=(3, 0), dst=(slice(2, 5), 0)),  # nodes 2..4 row 0
        ]
    )


def test_point_to_point_gather_to_same_dst_allowed():
    """Multiple point-to-point pipes whose dst is the same single node are
    allowed. A point-to-point gather uses cumulative semaphore waits at the
    receiver and does not use the collective transfer contract.
    """
    # Should not raise.
    ttl.PipeNet(
        [
            ttl.Pipe(src=(1, 0), dst=(0, 0)),
            ttl.Pipe(src=(2, 0), dst=(0, 0)),
            ttl.Pipe(src=(3, 0), dst=(0, 0)),
        ]
    )


def test_nonoverlapping_collective_pipes_in_pipenet_allowed():
    """Collective pipes targeting disjoint rectangles in the same PipeNet
    are allowed (e.g., per-row broadcasts)."""
    # Should not raise.
    ttl.PipeNet([ttl.Pipe(src=(0, r), dst=(slice(1, 4), r)) for r in range(3)])


def test_pipe_dst_slice_must_have_explicit_bounds():
    """ttl.Pipe rejects open slices in destination ranges."""
    with pytest.raises(ValueError, match="explicit start and stop"):
        ttl.Pipe(src=(0, 0), dst=(slice(None, 4), 0))
    with pytest.raises(ValueError, match="explicit start and stop"):
        ttl.Pipe(src=(0, 0), dst=(slice(0, None), 0))


def test_pipe_dst_slice_start_must_be_less_than_stop():
    with pytest.raises(ValueError, match="start must be < stop"):
        ttl.Pipe(src=(0, 0), dst=(slice(4, 4), 0))
    with pytest.raises(ValueError, match="start must be < stop"):
        ttl.Pipe(src=(0, 0), dst=(slice(4, 0), 0))


def test_pipe_dst_slice_step_must_be_one():
    """Strided collective destinations are not supported; non-1 step is lost by
    the inclusive-range lowering, so reject at construction."""
    with pytest.raises(ValueError, match="step must be 1 or None"):
        ttl.Pipe(src=(0, 0), dst=(slice(0, 4, 2), 0))
    with pytest.raises(ValueError, match="step must be 1 or None"):
        ttl.Pipe(src=(0, 0), dst=(0, slice(0, 4, 2)))
    # step == 1 is fine.
    ttl.Pipe(src=(0, 0), dst=(slice(0, 4, 1), 0))


def test_mixed_point_to_point_collective_in_one_pipenet_rejected():
    # Spec types `PipeNet[DstT](pipes: List[Pipe[DstT]])` so every pipe
    # shares one destination type; runtime validator pins the same rule.
    with pytest.raises(ValueError, match="may not mix point-to-point and collective"):
        ttl.PipeNet(
            [
                ttl.Pipe(src=(3, 0), dst=(0, 0)),
                ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0)),
            ]
        )


def test_all_point_to_point_pipenet_allowed():
    ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(1, 0)),
            ttl.Pipe(src=(0, 0), dst=(2, 0)),
        ]
    )


def test_all_collective_pipenet_allowed():
    ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0)),
            ttl.Pipe(src=(0, 1), dst=(slice(1, 3), 1)),
        ]
    )


def test_pipe_src_must_be_two_tuple():
    """`Pipe.src` is declared `Tuple[int, int]` on the hardware path; non-2
    lengths must be rejected at construction so users see the error at the
    source line rather than as a downstream emission failure.

    Sim accepts 1D coordinates (the `matmul_1d_mcast` example uses them),
    so this strict-2-tuple rule is hardware-only.
    """
    if not hasattr(ttl.Pipe, "_parse_dst"):
        pytest.skip("sim Pipe accepts 1D coords; rule is hardware-only")
    with pytest.raises(ValueError, match="src must be a 2-tuple"):
        ttl.Pipe(src=(0,), dst=(1, 0))
    with pytest.raises(ValueError, match="src must be a 2-tuple"):
        ttl.Pipe(src=(0, 1, 2), dst=(1, 0))
