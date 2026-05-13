# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the backend-neutral OperationPipeNets data type."""

import pytest

from ttl._pipenets import (
    NodeCoord,
    NodeRange,
    OperationPipeNets,
    PipeNetUse,
    PipeUse,
)


def _coord(*xs):
    return NodeCoord(coords=xs)


def _rng(lo, hi):
    return NodeRange(lo=tuple(lo), hi=tuple(hi))


class TestNodeRange:
    def test_rejects_mismatched_rank(self):
        with pytest.raises(ValueError, match="same rank"):
            NodeRange(lo=(0, 0), hi=(1,))

    def test_rejects_empty_axis(self):
        with pytest.raises(ValueError, match="lo < hi"):
            NodeRange(lo=(0, 0), hi=(1, 0))

    def test_accepts_minimal_range(self):
        rng = NodeRange(lo=(0, 0), hi=(1, 1))
        assert rng.lo == (0, 0)
        assert rng.hi == (1, 1)


class TestPipeNetIds:
    def test_pipenet_id_is_operation_local(self):
        graph = OperationPipeNets()
        first = graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(1, 0))])
        second = graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(0, 1))])
        assert first.id == 0
        assert second.id == 1
        assert isinstance(first, PipeNetUse)


class TestWorkExtent:
    def test_empty_graph_returns_none(self):
        assert OperationPipeNets().work_extent() is None

    def test_unicast_pipe_uses_max_coord_plus_one(self):
        graph = OperationPipeNets()
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(2, 3))])
        # extent must contain coords (0,0) and (2,3): rank-2, (3, 4).
        assert graph.work_extent() == (3, 4)

    def test_multicast_pipe_uses_hi_directly(self):
        graph = OperationPipeNets()
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_rng(lo=(1, 0), hi=(4, 1)))])
        # NodeRange.hi is exclusive, so it doubles as the extent.
        assert graph.work_extent() == (4, 1)

    def test_union_across_multiple_pipenets(self):
        graph = OperationPipeNets()
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(0, 5))])
        graph.add_pipe_net([PipeUse(src=_coord(3, 0), dst=_coord(3, 0))])
        assert graph.work_extent() == (4, 6)

    def test_mixed_rank_pipes_pad_with_one(self):
        graph = OperationPipeNets()
        # First pipe is rank-1, second rank-2: extent should be rank-2 with
        # the unspecified axis filled by 1.
        graph.add_pipe_net([PipeUse(src=_coord(2), dst=_coord(0))])
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(0, 3))])
        assert graph.work_extent() == (3, 4)


class TestValidate:
    def test_empty_graph_is_valid(self):
        OperationPipeNets().validate()

    def test_rejects_empty_pipenet(self):
        # `add_pipe_net` with no pipes is allowed for testing; validate catches it.
        graph = OperationPipeNets()
        graph.pipe_nets.append(PipeNetUse(id=0, pipes=()))
        with pytest.raises(ValueError, match="at least one pipe"):
            graph.validate()

    def test_rejects_overlapping_multicast_destinations(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(0, 0), dst=_rng(lo=(1, 0), hi=(4, 1))),
                PipeUse(src=_coord(0, 1), dst=_rng(lo=(2, 0), hi=(5, 1))),
            ]
        )
        with pytest.raises(ValueError, match="overlapping multicast destinations"):
            graph.validate()

    def test_unicast_gather_is_allowed(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(0, 0), dst=_coord(2, 2)),
                PipeUse(src=_coord(1, 0), dst=_coord(2, 2)),
            ]
        )
        graph.validate()  # no exception

    def test_disjoint_multicast_pipes_are_allowed(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(0, 0), dst=_rng(lo=(1, 0), hi=(3, 1))),
                PipeUse(src=_coord(0, 1), dst=_rng(lo=(3, 0), hi=(5, 1))),
            ]
        )
        graph.validate()  # no exception
