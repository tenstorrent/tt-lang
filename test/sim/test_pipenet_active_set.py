# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that the simulator skips kernel functions on nodes outside the PipeNet active set."""

import pytest

from python.sim import ttl, ttnn
from python.sim.context import get_context, reset_context
from python.sim.pipe import clear_pipe_net_registry, compute_active_linear_nodes


def _registry():
    return get_context().kernel_pipe_nets


def test_registry_starts_empty():
    reset_context()
    clear_pipe_net_registry()
    assert _registry() == []


def test_pipenet_registers_on_construction():
    reset_context()
    clear_pipe_net_registry()
    pipe = ttl.Pipe(src=(0, 0), dst=(0, 1))
    net = ttl.PipeNet([pipe])
    assert _registry() == [net]


def test_active_cores_includes_src_and_dst_range():
    clear_pipe_net_registry()
    # Multicast pipe: src=(0,0), dst x in [0,3], y=0.
    ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(0, 4), 0))])
    # On an 8x8 grid, row-major linearization gives (x, y) -> x*8 + y.
    active = compute_active_linear_nodes(grid=(8, 8))
    assert active is not None
    # Source (0,0) -> 0; destinations (0,0), (1,0), (2,0), (3,0) -> 0, 8, 16, 24.
    expected = {0, 8, 16, 24}
    assert active == expected


def test_active_cores_unions_multiple_pipenets():
    clear_pipe_net_registry()
    ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(0, 2), 0))])  # {0, 8}
    ttl.PipeNet([ttl.Pipe(src=(2, 0), dst=(2, slice(0, 2)))])  # {16, 17}
    active = compute_active_linear_nodes(grid=(8, 8))
    assert active == {0, 8, 16, 17}


def test_active_cores_unicast_single_dst():
    """Unicast pipe contributes the source cell and the single dst cell."""
    clear_pipe_net_registry()
    ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(2, 3))])
    active = compute_active_linear_nodes(grid=(8, 8))
    # src (0, 0) -> 0; dst (2, 3) -> 2*8 + 3 = 19.
    assert active == {0, 19}


def test_active_cores_none_when_no_pipenets():
    clear_pipe_net_registry()
    active = compute_active_linear_nodes(grid=(4, 4))
    assert active is None


def test_inactive_cores_skip_thread_bodies():
    """Run a multicast kernel on an 8x8 grid where pipes only touch nodes
    (0,0), (1,0), (2,0), (3,0). Verify only those nodes run the kernel functions.

    Per-node context is deep-copied by the simulator, so closure-captured
    Python sets cannot accumulate across nodes. We use ttl.trace events
    instead, which are emitted by the program scheduler before deep-copy.
    """
    clear_pipe_net_registry()
    reset_context()
    cfg = get_context().config
    cfg.trace_set = frozenset({"operation"})

    A = ttnn.rand((32, 32))
    O = ttnn.empty((32, 32))

    @ttl.operation(grid=(8, 8))
    def kernel(a: ttnn.Tensor, o: ttnn.Tensor):
        pipe = ttl.Pipe(src=(0, 0), dst=(slice(0, 4), 0))
        net = ttl.PipeNet([pipe])

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm0():
            pass

        @ttl.datamovement()
        def dm1():
            pass

    kernel(A, O)

    started = {
        ev.data["node"]
        for ev in get_context().trace_events
        if ev.event == "operation_start"
    }
    # On an 8x8 grid, nodes at (x=0..3, y=0) -> linear 0, 8, 16, 24.
    assert started == {0, 8, 16, 24}


def test_no_pipes_means_all_cores_run():
    """Without any PipeNet, the simulator must execute every node (legacy behavior)."""
    clear_pipe_net_registry()
    reset_context()
    cfg = get_context().config
    cfg.trace_set = frozenset({"operation"})

    A = ttnn.rand((32, 32))
    O = ttnn.empty((32, 32))

    @ttl.operation(grid=(2, 2))
    def kernel(a: ttnn.Tensor, o: ttnn.Tensor):
        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm0():
            pass

        @ttl.datamovement()
        def dm1():
            pass

    kernel(A, O)
    started = {
        ev.data["node"]
        for ev in get_context().trace_events
        if ev.event == "operation_start"
    }
    assert started == {0, 1, 2, 3}
