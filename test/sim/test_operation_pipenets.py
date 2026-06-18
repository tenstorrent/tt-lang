# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Python-only tests for the OperationPipeNets resource model."""

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


class TestActiveNodeSet:
    def test_empty_graph_returns_none(self):
        graph = OperationPipeNets()
        assert graph.active_node_set(grid=(8, 7)) is None

    def test_unicast_pipe_includes_src_and_dst(self):
        graph = OperationPipeNets()
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(2, 3))])
        # Row-major linearization: x * grid[1] + y on a (W, H) grid.
        # src (0,0) -> 0; dst (2,3) -> 2*7 + 3 = 17.
        assert graph.active_node_set(grid=(8, 7)) == {0, 17}

    def test_multicast_pipe_expands_destination_range(self):
        graph = OperationPipeNets()
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_rng(lo=(1, 0), hi=(4, 1)))])
        # src (0,0) -> 0; dsts (1..3, 0) -> 7, 14, 21 on grid (8, 7).
        assert graph.active_node_set(grid=(8, 7)) == {0, 7, 14, 21}

    def test_union_across_multiple_pipenets(self):
        graph = OperationPipeNets()
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(0, 1))])
        graph.add_pipe_net([PipeUse(src=_coord(1, 0), dst=_coord(1, 1))])
        # Linearized on grid (4, 4): 0, 1, 4, 5.
        assert graph.active_node_set(grid=(4, 4)) == {0, 1, 4, 5}


class TestPipeNetIds:
    def test_pipenet_id_is_operation_local(self):
        graph = OperationPipeNets()
        first = graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(1, 0))])
        second = graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(0, 1))])
        assert first.id == 0
        assert second.id == 1
        assert isinstance(first, PipeNetUse)


class TestValidate:
    def test_empty_graph_is_valid(self):
        OperationPipeNets().validate()

    def test_rejects_empty_pipenet(self):
        # `add_pipe_net` with no pipes is allowed for testing; validate catches it.
        graph = OperationPipeNets()
        graph.pipe_nets.append(PipeNetUse(id=0, pipes=()))
        with pytest.raises(ValueError, match="at least one pipe"):
            graph.validate()

    def test_overlapping_multicast_destinations_allowed(self):
        # Overlapping multicast destinations are supported via per-PipeNet
        # receiver counters; validation no longer rejects them.
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(0, 0), dst=_rng(lo=(1, 0), hi=(4, 1))),
                PipeUse(src=_coord(0, 1), dst=_rng(lo=(2, 0), hi=(5, 1))),
            ]
        )
        graph.validate()  # no exception

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

    def test_mixed_coord_ranks_rejected(self):
        # _linearize treats rank-1 coords as already-linear, so a rank-1 (5,)
        # and rank-2 (0, 5) on grid (8, 8) would alias to the same set element
        # in active_node_set. Reject the mix at the graph level.
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(0), dst=_coord(1)),
                PipeUse(src=_coord(0, 0), dst=_coord(1, 0)),
            ]
        )
        with pytest.raises(ValueError, match="coordinate ranks must be consistent"):
            graph.validate()


class TestPipeSyncSemaphores:
    def test_empty_graph_uses_no_pipe_resources(self):
        graph = OperationPipeNets()

        assert graph.num_pipe_sync_semaphores() == 0
        assert graph.num_pipe_global_semaphores() == 0

    def test_degenerate_multicast_uses_sram_address_table(self):
        graph = OperationPipeNets()
        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_rng((0, 0), (1, 1)))])

        assert graph.num_pipe_sync_semaphores() == 2
        assert graph.num_pipe_global_semaphores() == 0

    @pytest.mark.parametrize("recipient_count", [1, 2, 3, 50])
    def test_single_source_fanout_uses_constant_semaphore_count(self, recipient_count):
        node_count = recipient_count + 1
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [PipeUse(src=_coord(0, 0), dst=_rng((1, 0), (node_count, 1)))]
        )

        assert graph.num_pipe_sync_semaphores() == 2
        assert graph.num_pipe_global_semaphores() == 0

    def test_disjoint_non_loopback_multicast_reuses_source_local_ids(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(0, 0), dst=_rng((2, 0), (3, 1))),
                PipeUse(src=_coord(1, 0), dst=_rng((3, 0), (4, 1))),
            ]
        )

        assert graph.num_pipe_sync_semaphores() == 2
        assert graph.num_pipe_global_semaphores() == 0

    def test_overlapping_non_loopback_multicast_needs_one_ready_id_per_source_pipe(
        self,
    ):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(0, 0), dst=_rng((2, 0), (4, 1))),
                PipeUse(src=_coord(0, 0), dst=_rng((3, 0), (5, 1))),
            ]
        )

        assert graph.num_pipe_sync_semaphores() == 3
        assert graph.num_pipe_global_semaphores() == 0

    def test_same_source_pipes_use_global_ready_counters_when_local_ids_exhaust(
        self,
    ):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            PipeUse(src=_coord(0, 0), dst=_coord(dst_x, 0)) for dst_x in range(1, 17)
        )

        assert graph.num_pipe_sync_semaphores() == 1
        assert graph.num_pipe_global_semaphores() == 16

    def test_same_source_pipes_keep_local_ready_counters_at_limit(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            PipeUse(src=_coord(0, 0), dst=_coord(dst_x, 0)) for dst_x in range(1, 16)
        )

        assert graph.num_pipe_sync_semaphores() == 16
        assert graph.num_pipe_global_semaphores() == 0

    def test_multiple_pipenets_affect_global_ready_counter_threshold(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            PipeUse(src=_coord(0, 0), dst=_coord(dst_x, 0)) for dst_x in range(1, 15)
        )
        graph.add_pipe_net([PipeUse(src=_coord(1, 0), dst=_coord(2, 0))])

        assert graph.num_pipe_sync_semaphores() == 16
        assert graph.num_pipe_global_semaphores() == 0

        graph.add_pipe_net([PipeUse(src=_coord(0, 0), dst=_coord(15, 0))])

        assert graph.num_pipe_sync_semaphores() == 3
        assert graph.num_pipe_global_semaphores() == 16

    def test_global_ready_plan_counts_all_pipes(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            PipeUse(src=_coord(0, 0), dst=_coord(dst_x, 0)) for dst_x in range(1, 17)
        )
        graph.add_pipe_net(
            [
                PipeUse(src=_coord(1, 0), dst=_coord(18, 0)),
                PipeUse(src=_coord(2, 0), dst=_coord(19, 0)),
            ]
        )

        assert graph.num_pipe_sync_semaphores() == 2
        assert graph.num_pipe_global_semaphores() == 18

    def test_distinct_sources_keep_ready_counters_local(self):
        graph = OperationPipeNets()
        graph.add_pipe_net(
            PipeUse(src=_coord(src_x, 0), dst=_coord(0, 0)) for src_x in range(1, 17)
        )

        assert graph.num_pipe_sync_semaphores() == 2
        assert graph.num_pipe_global_semaphores() == 0

    def test_two_dimensional_all_to_all_multicast_count_is_constant(self):
        width = 32
        height = 16
        graph = OperationPipeNets()
        graph.add_pipe_net(
            PipeUse(src=_coord(src_x, src_y), dst=_rng((0, 0), (width, height)))
            for src_y in range(height)
            for src_x in range(width)
        )

        assert graph.num_pipe_sync_semaphores() == 2
        assert graph.num_pipe_global_semaphores() == 0
