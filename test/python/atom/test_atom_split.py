# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest %s -v

"""Off-device unit tests for the unified @ttl.operation thread splitter.

These drive ``split_function_body`` directly on small ASTs, so they need
neither ttnn nor a device and run anywhere ttl imports. They lock in the
split-time error paths for ambiguous thread ownership and the basic
compute/data-movement routing."""

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
        split_function_body(fn, dfb_param_names=set())


def test_tensor_backed_dfb_factory_and_publish_are_split_by_thread():
    fn = _fn(
        """
        def k(inp):
            inp_dfb = ttl.make_tensor_backed_dfb(inp, shape=(1, 1))
            inp_dfb.publish()
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
        local_dfb_names={"inp_dfb"},
    )

    trisc = _thread_src(result, "trisc")
    ncrisc = _thread_src(result, "ncrisc")
    brisc = _thread_src(result, "brisc")
    assert "make_tensor_backed_dfb" in trisc
    assert "make_tensor_backed_dfb" in ncrisc
    assert "make_tensor_backed_dfb" in brisc
    assert "inp_dfb.publish()" not in trisc
    assert "inp_dfb.publish()" in ncrisc
    assert "inp_dfb.publish()" not in brisc


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
    with pytest.raises(ValueError, match="multiple threads .*NCRISC, BRISC"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"a_cb"},
        )


def test_statement_mixing_compute_and_data_movement_is_rejected():
    """One statement cannot contain work for both TRISC and NCRISC."""
    fn = _fn(
        """
        def k(x, out):
            dst = out_cb.reserve()
            ttl.copy(ttl.exp(x), dst)
        """
    )

    with pytest.raises(ValueError, match="statement is pinned to multiple threads"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"out_cb"},
        )


def test_acquired_block_used_by_compute_and_data_movement_is_rejected():
    """A DFB acquire cannot be cloned onto compute and data movement."""
    fn = _fn(
        """
        def k(x, out):
            shared = out_cb.reserve()
            shared.store(x)
            ttl.copy(shared, out)
        """
    )

    with pytest.raises(ValueError, match="multiple threads .*TRISC, NCRISC"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"out_cb"},
        )


def test_with_acquired_block_used_by_multiple_threads_is_rejected():
    """The scoped DFB acquire form has the same single-thread requirement."""
    fn = _fn(
        """
        def k(x, out):
            with out_cb.reserve() as shared:
                shared.store(x)
                ttl.copy(shared, out)
        """
    )

    with pytest.raises(ValueError, match="acquire statement resolves to multiple"):
        split_function_body(
            fn,
            dfb_param_names=set(),
            local_dfb_names={"out_cb"},
        )


def test_assigned_copy_transfer_handle_is_rejected():
    """Assigned transfer handles remain unsupported until alias tracking lands."""
    fn = _fn(
        """
        def k(x, out):
            tx = ttl.copy(x, out)
            tx.wait()
        """
    )

    with pytest.raises(ValueError, match="assigned transfer handle"):
        split_function_body(
            fn,
            dfb_param_names=set(),
        )


def test_chained_copy_wait_routes_to_data_movement():
    """The supported non-assigned transfer wait remains on NCRISC."""
    fn = _fn(
        """
        def k(x, out):
            ttl.copy(x, out).wait()
        """
    )

    result = split_function_body(
        fn,
        dfb_param_names=set(),
    )

    assert "ttl.copy" not in _thread_src(result, "trisc")
    assert "ttl.copy" in _thread_src(result, "ncrisc")
    assert "ttl.copy" not in _thread_src(result, "brisc")


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
