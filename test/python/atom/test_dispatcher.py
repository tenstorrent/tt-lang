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


@ttl.operation(grid=(1, 1))
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


def test_dispatcher_records_targets_by_stable_operation_identity():
    targets = _dispatcher.dispatch_targets

    assert tuple(targets.values()) == (_dispatch_child,)
    assert tuple(targets) == (_dispatch_child._spec.operation_identity,)
    with pytest.raises(TypeError):
        targets["replacement"] = _dispatch_child


def test_dispatcher_ast_is_not_mutable_through_public_property():
    dispatcher_ast = _dispatcher.dispatcher_ast
    dispatcher_ast.body.clear()

    assert _dispatcher.dispatcher_ast.body


def test_ordinary_operation_has_no_dispatcher_ast():
    assert not _dispatch_child.dispatcher
    assert _dispatch_child.dispatcher_ast is None
    assert not _dispatch_child.dispatch_targets


def test_dispatcher_excludes_target_in_statically_disabled_branch():
    @ttl.operation(grid=(1, 1))
    def disabled_child():
        disabled_value = 1

    enabled = False

    @ttl.operation(dispatcher=True)
    def dispatcher():
        _dispatch_child()
        if enabled:
            disabled_child()

    assert tuple(dispatcher.dispatch_targets.values()) == (_dispatch_child,)


def test_dispatcher_cannot_use_ordinary_compile_path():
    @ttl.operation(grid=(1, 1), dispatcher=True)
    def dispatcher():
        _dispatch_child()

    with pytest.raises(
        ValueError,
        match="dispatcher 'dispatcher' is control-only",
    ):
        dispatcher()


def test_dispatcher_target_must_compile_independently():
    @ttl.operation()
    def expand_only_child():
        child_value = 1

    with pytest.raises(
        ValueError,
        match="dispatcher target 'expand_only_child' must declare a grid",
    ):

        @ttl.operation(dispatcher=True)
        def invalid_dispatcher():
            expand_only_child()


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
