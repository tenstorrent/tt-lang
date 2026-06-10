# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Off-device unit tests for the @ttl.atom thread splitter.

These drive ``split_function_body`` directly on small ASTs, so they need
neither ttnn nor a device and run anywhere ttl imports. They lock in the
split-time error paths (unknown op, no-use producer, NCRISC/BRISC
double-reserve) and the basic compute/data-movement routing."""

import ast
import textwrap

import pytest

from ttl._src.atom_split import split_function_body


def _fn(src: str) -> ast.FunctionDef:
    return ast.parse(textwrap.dedent(src)).body[0]


def _thread_src(result, thread: str) -> str:
    return "\n".join(ast.unparse(s) for s in result.body_for(thread))


def test_unknown_ttl_op_is_rejected():
    fn = _fn(
        """
        def k(a):
            ttl.frobnicate(a)
        """
    )
    with pytest.raises(ValueError, match="unknown ttl.frobnicate"):
        split_function_body(fn, dfb_param_names=set(), all_param_names={"a"})


def test_producer_with_no_uses_is_rejected():
    fn = _fn(
        """
        def k():
            blk = a_cb.wait()
        """
    )
    with pytest.raises(ValueError, match="has no uses"):
        split_function_body(fn, dfb_param_names=set(), local_dfb_names={"a_cb"})


def test_producer_split_across_ncrisc_and_brisc_is_rejected():
    """A single reserve whose block feeds both an if_src (BRISC) and an
    if_dst (NCRISC) callback would double-reserve the CB."""
    fn = _fn(
        """
        def k(net):
            shared = a_cb.reserve()

            def send(pipe):
                ttl.copy(shared, pipe)

            net.if_src(send)

            def recv(pipe):
                ttl.copy(pipe, shared)

            net.if_dst(recv)
        """
    )
    with pytest.raises(ValueError, match="both NCRISC"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            all_param_names={"net"},
            local_dfb_names={"a_cb"},
        )


def test_compute_and_dm_route_to_separate_threads():
    """Copies land on NCRISC, the compute op on TRISC, and the unused
    BRISC thread is empty."""
    fn = _fn(
        """
        def k(a, out):
            a_blk = a_cb.reserve()
            ttl.copy(a, a_blk)
            s = out_cb.reserve()
            x = a_cb.wait()
            s.store(ttl.exp(x))
            done = out_cb.wait()
            ttl.copy(done, out)
        """
    )
    result = split_function_body(
        fn,
        dfb_param_names=set(),
        all_param_names={"a", "out"},
        local_dfb_names={"a_cb", "out_cb"},
    )

    trisc = _thread_src(result, "trisc")
    ncrisc = _thread_src(result, "ncrisc")
    brisc = _thread_src(result, "brisc")

    assert "ttl.exp" in trisc
    assert "ttl.copy" not in trisc
    assert "ttl.copy" in ncrisc
    assert "ttl.exp" not in ncrisc
    assert brisc.strip() == "pass"
