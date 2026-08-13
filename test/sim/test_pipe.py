# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for sim PipeNet predicates: is_active, is_src, is_dst."""

from __future__ import annotations

import pytest
import torch

from test_utils import make_zeros_tensor

from sim import ttl, ttnn
from sim.context import get_context
from sim.nodecontext import pipe_crosses_mesh
from sim.trace import ALL_CATEGORIES, set_tracing


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


class TestPipeCrossesMesh:
    """pipe_crosses_mesh classifies a pipe as fabric (cross-device) vs on-chip NoC.

    A grid's leading ``len(grid) - 2`` dims are device-mesh axes; the trailing two
    are the Tensix core grid.  A pipe is fabric iff its source and any destination
    differ on a mesh axis.
    """

    def test_rank2_grid_never_fabric(self) -> None:
        """A 2D grid has no mesh axes: every pipe is on-chip."""
        assert pipe_crosses_mesh((0, 0), (1, 1), (4, 4)) is False

    def test_cross_card_unicast_is_fabric(self) -> None:
        """4D grid, endpoints differ on a mesh axis (card column) -> fabric."""
        # grid (4, 8, 13, 10): mesh axes (4, 8). src/dst differ on axis 1.
        assert pipe_crosses_mesh((0, 0, 0, 5), (0, 1, 0, 5), (4, 8, 13, 10)) is True

    def test_on_chip_unicast_not_fabric(self) -> None:
        """4D grid, endpoints share mesh coord, differ only on core dims -> NoC."""
        assert pipe_crosses_mesh((0, 1, 0, 5), (0, 1, 12, 5), (4, 8, 13, 10)) is False

    def test_on_chip_multicast_not_fabric(self) -> None:
        """A multicast slice on a core axis (same mesh coord) stays on-chip."""
        # mcast across the cx core axis (index 2); mesh dims (0, 1) fixed == src.
        assert (
            pipe_crosses_mesh((0, 1, 0, 5), (0, 1, slice(0, 13), 5), (4, 8, 13, 10))
            is False
        )

    def test_multicast_spanning_mesh_axis_is_fabric(self) -> None:
        """A slice on a mesh axis that covers other devices -> fabric."""
        assert (
            pipe_crosses_mesh((0, 0, 0, 5), (0, slice(0, 8), 0, 5), (4, 8, 13, 10))
            is True
        )

    def test_ring_wraparound_is_fabric(self) -> None:
        """A 3D ring step (single mesh axis) to a different card is fabric."""
        # grid (2, 1, 1): mesh axis (2,). 0 -> 1 differs on the mesh axis.
        assert pipe_crosses_mesh((0, 0, 0), (1, 0, 0), (2, 1, 1)) is True

    def test_linear_index_is_unflattened(self) -> None:
        """A bare linear index is unambiguous: it is unflattened, not rejected."""
        # grid (2, 1, 1): mesh axis (2,), 2 devices, 1 core each -> linear == device.
        # Linear 0 and 1 land on different cards, so the pipe is fabric.
        assert pipe_crosses_mesh(0, 1, (2, 1, 1)) is True
        assert pipe_crosses_mesh((0,), (1,), (2, 1, 1)) is True
        # Both on the same (only) card of a single-mesh-axis grid -> on-chip.
        assert pipe_crosses_mesh(0, 0, (2, 1, 1)) is False

    def test_multi_element_sub_rank_coord_raises(self) -> None:
        """A multi-element coordinate shorter than grid rank is ambiguous."""
        # grid rank 4 but endpoints are 2-tuples: node() flattens leading axes
        # while flatten_node_index() flattens differently, so the entries cannot
        # be mapped one-to-one to mesh axes.
        with pytest.raises(ValueError, match="use a full-rank coordinate"):
            pipe_crosses_mesh((0, 0), (1, 0), (4, 8, 13, 10))


class TestPipeIsFabricNeverRaises:
    """The trace-only ``fabric`` classifier degrades instead of raising.

    ``pipe_crosses_mesh`` rejects an ambiguous multi-element sub-rank coordinate,
    but that classification only annotates trace events, so an otherwise-valid
    traced run must not crash on it.
    """

    def test_ambiguous_endpoint_reported_non_fabric(self) -> None:
        from sim.copyhandlers import _pipe_is_fabric
        from sim.greenlet_scheduler import GreenletScheduler

        # Rank-2 endpoints on a rank-4 grid: neither a linear index nor full rank.
        pipe = ttl.Pipe((0, 0), (1, 0))
        ctx = get_context()
        scheduler = GreenletScheduler()
        scheduler.grid = (2, 2, 1, 1)
        ctx.scheduler = scheduler
        try:
            with pytest.raises(ValueError, match="use a full-rank coordinate"):
                pipe_crosses_mesh(pipe.src, pipe.dst, scheduler.grid)
            assert _pipe_is_fabric(pipe) is False
        finally:
            ctx.scheduler = None

    def test_no_scheduler_grid_is_non_fabric(self) -> None:
        """With no launch grid recorded there are no mesh axes to cross."""
        from sim.copyhandlers import _pipe_is_fabric

        ctx = get_context()
        ctx.scheduler = None
        assert _pipe_is_fabric(ttl.Pipe((0, 0), (1, 0))) is False


class TestCrossCardPipe:
    """End-to-end cross-device (fabric) pipe transfer on a grid with mesh axes.

    On grid ``(2, 1, 1)`` the leading dim is a 2-device mesh axis; a pipe from
    node ``(0,0,0)`` to ``(1,0,0)`` crosses it.  The functional model routes the
    payload through the shared pipe buffer regardless of device, so the
    destination device receives the source device's tile.
    """

    def _run(self, *, capture_trace: bool = False) -> tuple[ttnn.Tensor, list]:
        ctx = get_context()
        if capture_trace:
            set_tracing(ALL_CATEGORIES)

        inp = ttnn.from_torch(torch.full((32, 32), 7.0))
        out = ttnn.from_torch(torch.zeros(32, 32))

        @ttl.operation(grid=(2, 1, 1))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            net = ttl.PipeNet([ttl.Pipe((0, 0, 0), (1, 0, 0))])
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1))
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1))

            @ttl.compute()
            def compute() -> None:
                pass

            @ttl.datamovement()
            def dm_send() -> None:
                def _send(pipe_id):
                    with dfb.reserve() as blk:
                        tx = ttl.copy(a[0, 0], blk)
                        tx.wait()
                    with dfb.wait() as blk:
                        tx = ttl.copy(blk, pipe_id)
                        tx.wait()

                net.if_src(_send)

            @ttl.datamovement()
            def dm_recv() -> None:
                def _recv(pipe_id):
                    with out_dfb.reserve() as blk:
                        tx = ttl.copy(pipe_id, blk)
                        tx.wait()
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o[0, 0])
                        tx.wait()

                net.if_dst(_recv)

        kernel(inp, out)
        return out, list(ctx.trace_events)

    def test_cross_mesh_pipe_transfers_data(self) -> None:
        """The destination device receives the source device's tile across the mesh."""
        out, _ = self._run()
        assert torch.allclose(out.to_torch(), torch.full((32, 32), 7.0))

    def test_cross_mesh_pipe_marked_fabric_in_trace(self) -> None:
        """pipe_send / pipe_recv across the mesh axis carry fabric=True."""
        _, events = self._run(capture_trace=True)
        pipe_events = [e for e in events if e.event in ("pipe_send", "pipe_recv")]
        assert pipe_events, "expected pipe_send/pipe_recv events"
        for ev in pipe_events:
            assert ev.data.get("fabric") is True, f"expected fabric=True in {ev}"


class TestOnChipPipeClassification:
    """An on-chip pipe (same device, differing only on core dims) is not fabric."""

    def test_on_chip_pipe_not_fabric_in_trace(self) -> None:
        ctx = get_context()
        set_tracing(ALL_CATEGORIES)

        inp = ttnn.from_torch(torch.full((32, 32), 3.0))
        out = ttnn.from_torch(torch.zeros(32, 32))

        # grid (1, 1, 2): a single device (mesh axis size 1) with a 1x2 core grid;
        # the pipe stays within the device, differing only on a core dim.
        @ttl.operation(grid=(1, 1, 2))
        def kernel(a: ttnn.Tensor, o: ttnn.Tensor) -> None:
            net = ttl.PipeNet([ttl.Pipe((0, 0, 0), (0, 0, 1))])
            dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1))
            out_dfb = ttl.make_dataflow_buffer_like(o, shape=(1, 1))

            @ttl.compute()
            def compute() -> None:
                pass

            @ttl.datamovement()
            def dm_send() -> None:
                def _send(pipe_id):
                    with dfb.reserve() as blk:
                        tx = ttl.copy(a[0, 0], blk)
                        tx.wait()
                    with dfb.wait() as blk:
                        tx = ttl.copy(blk, pipe_id)
                        tx.wait()

                net.if_src(_send)

            @ttl.datamovement()
            def dm_recv() -> None:
                def _recv(pipe_id):
                    with out_dfb.reserve() as blk:
                        tx = ttl.copy(pipe_id, blk)
                        tx.wait()
                    with out_dfb.wait() as blk:
                        tx = ttl.copy(blk, o[0, 0])
                        tx.wait()

                net.if_dst(_recv)

        kernel(inp, out)
        pipe_events = [
            e for e in ctx.trace_events if e.event in ("pipe_send", "pipe_recv")
        ]
        assert pipe_events, "expected pipe_send/pipe_recv events"
        for ev in pipe_events:
            assert ev.data.get("fabric") is False, f"expected fabric=False in {ev}"
