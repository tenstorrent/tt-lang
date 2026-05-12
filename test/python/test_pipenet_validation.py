# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Negative tests for ttl.PipeNet construction-time validation.

Pure Python tests: no device, no MLIR. Pin the user-visible error
contract for invalid PipeNet shapes.
"""

# REQUIRES: ttnn
# RUN: %python -m pytest %s -v

import pytest
import ttl


def test_empty_pipenet_rejected():
    with pytest.raises(ValueError, match="at least one pipe"):
        ttl.PipeNet([])


def test_within_pipenet_overlapping_mcast_dst_rejected():
    """Two multicast pipes whose destination rectangles intersect inside a
    single PipeNet must be rejected at construction time.

    The two pipes both target column 1 rows 0..3, so the node at (1, 1)
    would receive from both. PipeNet's single shared semaphore pair
    cannot disambiguate, so this is rejected (issue #505).
    """
    with pytest.raises(ValueError, match="overlapping multicast destinations"):
        ttl.PipeNet(
            [
                ttl.Pipe(src=(0, 0), dst=(1, slice(0, 4))),
                ttl.Pipe(src=(2, 0), dst=(1, slice(0, 4))),
            ]
        )


def test_within_pipenet_partially_overlapping_mcast_dst_rejected():
    """Multicast destinations that overlap on even one node are rejected."""
    with pytest.raises(ValueError, match="overlapping multicast destinations"):
        ttl.PipeNet(
            [
                ttl.Pipe(src=(0, 0), dst=(slice(0, 3), 0)),  # nodes 0..2 row 0
                ttl.Pipe(src=(3, 0), dst=(slice(2, 5), 0)),  # nodes 2..4 row 0
            ]
        )


def test_unicast_gather_to_same_dst_allowed():
    """Multiple unicast pipes whose dst is the same single node are
    allowed: a unicast gather uses cumulative semaphore waits at the
    receiver, not the multicast handshake. The within-PipeNet rule
    rejects only multicast overlap.
    """
    # Should not raise.
    ttl.PipeNet(
        [
            ttl.Pipe(src=(1, 0), dst=(0, 0)),
            ttl.Pipe(src=(2, 0), dst=(0, 0)),
            ttl.Pipe(src=(3, 0), dst=(0, 0)),
        ]
    )


def test_nonoverlapping_mcast_pipes_in_pipenet_allowed():
    """Multicast pipes targeting disjoint rectangles in the same PipeNet
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


def test_mixed_unicast_multicast_in_one_pipenet_rejected():
    # Spec types `PipeNet[DstT](pipes: List[Pipe[DstT]])` so every pipe
    # shares one destination type; runtime validator pins the same rule.
    with pytest.raises(ValueError, match="may not mix unicast and multicast"):
        ttl.PipeNet(
            [
                ttl.Pipe(src=(3, 0), dst=(0, 0)),
                ttl.Pipe(src=(0, 0), dst=(slice(1, 3), 0)),
            ]
        )


def test_all_unicast_pipenet_allowed():
    ttl.PipeNet(
        [
            ttl.Pipe(src=(0, 0), dst=(1, 0)),
            ttl.Pipe(src=(0, 0), dst=(2, 0)),
        ]
    )


def test_all_multicast_pipenet_allowed():
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
