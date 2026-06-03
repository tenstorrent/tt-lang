# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for sim PipeNet predicates: is_active, is_src, is_dst."""

from __future__ import annotations

import pytest

from test_utils import make_zeros_tensor

from sim import ttl, ttnn


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
