# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for sim PipeNet predicates: is_active, is_src, is_dst."""

from __future__ import annotations

from typing import cast

import pytest

from test_utils import make_ones_tensor, make_zeros_tensor

from sim import copy, ttl, ttnn
from sim.dfb import Block
from sim.pipe import build_pipenets
from sim.program import _dedupe_pipe_nets  # type: ignore[reportPrivateUsage]

# A net belonging to no operation in this file, standing in for one declared for a
# neighbouring operation in the same module. Read by
# test_a_net_the_operation_never_mentions_is_not_its_net, whose operation must not
# pick it up.
_NET_OF_ANOTHER_OPERATION = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(0, 1))])


class TestPipeNetPredicates:
    """PipeNet.is_src / is_dst / is_active use ttl.node(); run inside @ttl.operation."""

    def test_unicast_src_dst_inactive(self) -> None:
        """Unicast (0,0) -> (1,0): only those two nodes participate on a 2x2 grid."""
        pipe = ttl.Pipe((0, 0), (1, 0))
        net = ttl.PipeNet([pipe])

        @ttl.operation(grid=(2, 2))
        def op(a: ttnn.Tensor, b: ttnn.Tensor) -> None:
            @ttl.compute()
            def compute() -> None:
                cid = ttl.node(dims=1)
                if cid == 0:
                    assert net.is_src() is True
                    assert net.is_dst() is False
                    assert net.is_active() is True
                elif cid == 2:
                    assert net.is_src() is False
                    assert net.is_dst() is True
                    assert net.is_active() is True
                elif cid in (1, 3):
                    assert net.is_src() is False
                    assert net.is_dst() is False
                    assert net.is_active() is False
                else:
                    raise AssertionError(f"unexpected node {cid}")

            @ttl.datamovement()
            def dm0() -> None:
                pass

            @ttl.datamovement()
            def dm1() -> None:
                pass

        x = make_zeros_tensor(32, 32)
        op(x, x)

    def test_multicast_src_and_multiple_dst(self) -> None:
        """Multicast from (0,0) to columns 1..2 on row 0; grid 2x3."""
        net = ttl.PipeNet([ttl.Pipe((0, 0), (0, slice(1, 3)))])

        @ttl.operation(grid=(2, 3))
        def op(a: ttnn.Tensor, b: ttnn.Tensor) -> None:
            @ttl.compute()
            def compute() -> None:
                cid = ttl.node(dims=1)
                if cid == 0:
                    assert net.is_src() and not net.is_dst() and net.is_active()
                elif cid in (1, 2):
                    assert not net.is_src() and net.is_dst() and net.is_active()
                elif cid in (3, 4, 5):
                    assert not net.is_src() and not net.is_dst() and not net.is_active()
                else:
                    raise AssertionError(f"unexpected node {cid}")

            @ttl.datamovement()
            def dm0() -> None:
                pass

            @ttl.datamovement()
            def dm1() -> None:
                pass

        x = make_zeros_tensor(32, 32)
        op(x, x)

    def test_two_pipes_union_for_is_active(self) -> None:
        """Two unicasts in one net: (0,0)->(0,1) and (1,0)->(1,1)."""
        net = ttl.PipeNet(
            [
                ttl.Pipe((0, 0), (0, 1)),
                ttl.Pipe((1, 0), (1, 1)),
            ]
        )

        @ttl.operation(grid=(2, 2))
        def op(a: ttnn.Tensor, b: ttnn.Tensor) -> None:
            @ttl.compute()
            def compute() -> None:
                cid = ttl.node(dims=1)
                if cid == 0:
                    assert net.is_src() and not net.is_dst() and net.is_active()
                elif cid == 1:
                    assert not net.is_src() and net.is_dst() and net.is_active()
                elif cid == 2:
                    assert net.is_src() and not net.is_dst() and net.is_active()
                elif cid == 3:
                    assert not net.is_src() and net.is_dst() and net.is_active()
                else:
                    raise AssertionError(f"unexpected node {cid}")

            @ttl.datamovement()
            def dm0() -> None:
                pass

            @ttl.datamovement()
            def dm1() -> None:
                pass

        x = make_zeros_tensor(32, 32)
        op(x, x)

    def test_is_src_and_is_dst_disjoint_roles_unicast(self) -> None:
        """On a unicast edge, source is not dst and destination is not src."""
        net = ttl.PipeNet([ttl.Pipe((0, 0), (0, 1))])

        @ttl.operation(grid=(1, 2))
        def op(a: ttnn.Tensor, b: ttnn.Tensor) -> None:
            @ttl.compute()
            def compute() -> None:
                cid = ttl.node(dims=1)
                if cid == 0:
                    assert net.is_src() and not net.is_dst()
                elif cid == 1:
                    assert not net.is_src() and net.is_dst()
                else:
                    raise AssertionError(f"unexpected node {cid}")

            @ttl.datamovement()
            def dm0() -> None:
                pass

            @ttl.datamovement()
            def dm1() -> None:
                pass

        x = make_zeros_tensor(32, 32)
        op(x, x)


class TestPipeNetDiscovery:
    """Aggregation of the PipeNets the per-node body runs construct.

    The body is re-run once per node, so each run builds its own PipeNet
    object; these pin what the operation's graph ends up holding.
    """

    def test_identical_per_node_nets_collapse_to_one(self) -> None:
        """A node-independent net is one net, however many nodes built it."""
        nets = [ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))]) for _ in range(4)]

        graph = build_pipenets(_dedupe_pipe_nets(nets))

        assert [net.id for net in graph.pipe_nets] == [0]
        assert graph.active_node_set((2, 2)) == {0, 2}

    def test_node_dependent_nets_are_kept_and_their_active_sets_union(self) -> None:
        """A net whose pipes vary per node contributes one entry per version.

        Each version is validated on its own and every one is active, so a node
        runs when it participates in any of them.
        """
        nets = [ttl.PipeNet([ttl.Pipe(src=(0, n), dst=(1, n))]) for n in range(2)]

        graph = build_pipenets(_dedupe_pipe_nets(nets))
        graph.validate()

        assert [net.id for net in graph.pipe_nets] == [0, 1]
        assert graph.active_node_set((2, 2)) == {0, 1, 2, 3}

    def test_point_to_point_and_collective_nets_coexist(self) -> None:
        """An operation may declare both kinds of net.

        Point-to-point and collective pipes may not be mixed within one net,
        but nothing stops an operation from declaring one net of each.
        """
        p2p = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(1, 0))])
        collective = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(0, 2), 1))])

        graph = build_pipenets(_dedupe_pipe_nets([p2p, collective]))
        graph.validate()

        assert [net.id for net in graph.pipe_nets] == [0, 1]
        assert graph.active_node_set((2, 2)) == {0, 1, 2, 3}

    def test_pipes_are_readable_without_reaching_for_the_private_field(self) -> None:
        """PipeNet.pipes reports the declared pipes, as the compiler's does."""
        pipes = [ttl.Pipe(src=(0, 0), dst=(1, 0)), ttl.Pipe(src=(0, 1), dst=(1, 1))]

        assert ttl.PipeNet(pipes).pipes == tuple(pipes)

    def test_objects_that_are_not_pipe_nets_do_not_merge(self) -> None:
        """Dedupe keys on the pipes themselves, with nothing to default to.

        A defaulted lookup would give every net-less object the same key and
        collapse unrelated entries into one without complaint.  The complaint has
        to name ``pipes``, since an attribute error about any other name means the
        key is being read off the wrong attribute -- which no real net would
        answer either.
        """

        class NotANet:
            pass

        with pytest.raises(AttributeError, match="pipes"):
            _dedupe_pipe_nets([NotANet(), NotANet()])  # type: ignore[list-item]

    def test_a_net_the_operation_never_mentions_is_not_its_net(self) -> None:
        """Another operation's module-level net does not shrink this one's nodes.

        A net is this operation's when its body or kernels refer to it, which is
        what the specification means by captured from an enclosing scope ("Pipe
        net"). Every net an operation holds narrows which nodes run its kernels, so
        taking a net that belongs to a neighbouring operation in the same file
        leaves nodes with work unrun -- and with no pipe code to look at, the only
        symptom is a partly written output.
        """

        @ttl.operation(grid=(2, 2))
        def op(out: ttnn.Tensor) -> None:
            out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=1)

            @ttl.compute()
            def compute() -> None:
                block = out_dfb.reserve()
                block.store(Block.from_tensor(make_ones_tensor(32, 32)))
                block.push()

            @ttl.datamovement()
            def dm0() -> None:
                pass

            @ttl.datamovement()
            def dm1() -> None:
                row, column = cast(tuple[int, int], ttl.node(dims=2))
                block = out_dfb.wait()
                copy(block, out[row : row + 1, column : column + 1]).wait()
                block.pop()

        out = make_zeros_tensor(64, 64)
        op(out)

        # _NET_OF_ANOTHER_OPERATION covers nodes (0, 0) and (0, 1) only, so a
        # discovery that picked it up would leave the bottom two tiles at zero.
        assert (out.to_torch() == 1).all(), "some node did not run its kernels"


class TestPipeDstSliceValidation:
    """Construction-time validation of `dst` slices in sim ttl.Pipe.
    Must stay in lockstep with compiler-side validation in python/ttl/pipe.py."""

    def test_step_must_be_one_or_none(self) -> None:
        with pytest.raises(ValueError, match="step must be 1 or None"):
            ttl.Pipe(src=(0, 0), dst=(slice(0, 4, 2), 0))
        with pytest.raises(ValueError, match="step must be 1 or None"):
            ttl.Pipe(src=(0, 0), dst=(0, slice(0, 4, 2)))
        ttl.Pipe(src=(0, 0), dst=(slice(0, 4, 1), 0))
        ttl.Pipe(src=(0, 0), dst=(slice(0, 4), 0))
