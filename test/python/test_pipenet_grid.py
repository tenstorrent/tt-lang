# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# RUN: %python -m pytest %s -v

"""Registry-side-effect tests for PipeNet.

The active-set guard (issue #541) is driven by a registry that captures every
PipeNet constructed during a kernel trace. Active set computation runs over
this registry; these tests cover the registration and clear-on-trace
side effects without requiring a full kernel compile.
"""

import ttl
from ttl.pipe import _clear_pipe_net_registry, _pipe_net_registry


def test_pipenet_registers_on_construction():
    _clear_pipe_net_registry()
    pipes = [ttl.Pipe(src=(0, 0), dst=(slice(0, 4), 0))]
    net = ttl.PipeNet(pipes)
    assert _pipe_net_registry == [net]


def test_pipenet_registers_multiple_in_order():
    _clear_pipe_net_registry()
    net_a = ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(0, 2), 0))])
    net_b = ttl.PipeNet([ttl.Pipe(src=(0, 1), dst=(slice(0, 4), 1))])
    assert _pipe_net_registry == [net_a, net_b]


def test_clear_pipe_net_registry_empties_it():
    ttl.PipeNet([ttl.Pipe(src=(0, 0), dst=(slice(0, 2), 0))])
    assert len(_pipe_net_registry) >= 1
    _clear_pipe_net_registry()
    assert _pipe_net_registry == []


def test_pipenet_still_rejects_empty_pipes():
    import pytest
    _clear_pipe_net_registry()
    with pytest.raises(ValueError, match="at least one pipe"):
        ttl.PipeNet([])
