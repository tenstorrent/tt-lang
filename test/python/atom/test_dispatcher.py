# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Frontend contracts for static dispatch controller operations."""

import ast

import pytest

import ttl


@ttl.operation()
def _dispatch_child():
    child_value = 1


_ENABLE_CHILD = True


@ttl.operation(dispatcher=True)
def _dispatcher():
    if _ENABLE_CHILD:
        _dispatch_child()


def test_dispatcher_preserves_pre_inline_operation_calls():
    dispatcher_source = ast.unparse(_dispatcher.dispatcher_ast)

    assert _dispatcher.dispatcher
    assert _dispatcher._ttl_operation_kind == "dispatcher"
    assert "_dispatch_child()" in dispatcher_source
    assert "if _ENABLE_CHILD" not in dispatcher_source
    assert "_dispatch_child()" not in _dispatcher._spec.source


def test_dispatcher_ast_is_not_mutable_through_public_property():
    dispatcher_ast = _dispatcher.dispatcher_ast
    dispatcher_ast.body.clear()

    assert _dispatcher.dispatcher_ast.body


def test_ordinary_operation_has_no_dispatcher_ast():
    assert not _dispatch_child.dispatcher
    assert _dispatch_child.dispatcher_ast is None


def test_dispatcher_rejects_explicit_multi_kernel_body():
    with pytest.raises(
        ValueError,
        match="dispatcher=True requires a unified operation body",
    ):

        @ttl.operation(dispatcher=True)
        def invalid_dispatcher():
            @ttl.compute()
            def compute():
                pass


def test_dispatcher_cannot_be_composed_into_an_operation():
    with pytest.raises(
        ValueError,
        match="cannot compose dispatcher operation '_dispatcher'",
    ):

        @ttl.operation()
        def invalid_composition():
            _dispatcher()


def test_dispatcher_option_requires_bool():
    with pytest.raises(TypeError, match="dispatcher must be a bool"):
        ttl.operation(dispatcher=1)
