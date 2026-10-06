# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

"""Frontend contracts for static dispatch controller operations."""

import ast

import pytest
import torch

import ttl
import ttnn
from ttl.ir import MLIRError


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


@ttl.operation(
    grid=(1, 1),
    dispatch_arguments={
        "source": ttl.DispatchArgument("read", "handoff"),
        "destination": ttl.DispatchArgument("write", "handoff"),
    },
)
def _dispatch_write(source, destination):
    pass


@ttl.operation(
    grid=(1, 1),
    dispatch_arguments={
        "source": ttl.DispatchArgument("read", "handoff"),
        "destination": ttl.DispatchArgument("write", "handoff"),
    },
)
def _dispatch_transform(source, destination):
    pass


@ttl.operation(dispatcher=True)
def _emitted_dispatcher(source, intermediate, destination):
    _dispatch_write(source, intermediate)
    _dispatch_transform(source=intermediate, destination=destination)
    _dispatch_write(destination, intermediate)


def _host_tensor():
    return ttnn.Tensor(torch.ones((1, 1, 32, 32), dtype=torch.bfloat16))


def test_dispatcher_emits_targets_and_ordered_invocations():
    module = _emitted_dispatcher.emit_dispatch_ir(
        _host_tensor(), _host_tensor(), _host_tensor()
    )
    text = str(module)

    assert text.count("ttl.dispatch.target") == 2
    assert text.count("ttl.dispatch.invoke") == 3
    assert "func.func @_emitted_dispatcher" in text
    assert "ttl.dispatcher" in text
    first = text.index("ttl.dispatch.invoke @_dispatch_write")
    second = text.index("ttl.dispatch.invoke @_dispatch_transform")
    third = text.index("ttl.dispatch.invoke @_dispatch_write", first + 1)
    assert first < second < third
    assert "ttl.dispatch.invoke @_dispatch_write(%arg0, %arg1 :" in text
    assert "ttl.dispatch.invoke @_dispatch_transform(%arg1, %arg2 :" in text
    assert "ttl.dispatch.invoke @_dispatch_write(%arg2, %arg1 :" in text
    assert _dispatch_write._spec.operation_identity in text
    assert _dispatch_transform._spec.operation_identity in text


def test_dispatcher_runs_dedicated_resolution_pipeline():
    module = _emitted_dispatcher.resolve_dispatch_ir(
        _host_tensor(), _host_tensor(), _host_tensor()
    )
    text = str(module)

    assert "attributes {ttl.dispatch.resolved, ttl.dispatcher}" in text
    first = text.index("ttl.dispatch.invoke @_dispatch_write")
    second = text.index("ttl.dispatch.invoke @_dispatch_transform")
    third = text.index("ttl.dispatch.invoke @_dispatch_write", first + 1)
    assert first < second < third


def test_resolved_dispatcher_exposes_backend_neutral_schedule():
    resolved = _emitted_dispatcher.resolve_dispatch(
        _host_tensor(), _host_tensor(), _host_tensor()
    )

    assert isinstance(resolved, ttl.ResolvedDispatcher)
    assert resolved.name == "_emitted_dispatcher"
    assert len(resolved.argument_types) == 3
    assert tuple(target.symbol for target in resolved.targets) == (
        "_dispatch_write",
        "_dispatch_transform",
    )
    assert tuple(target.argument_names for target in resolved.targets) == (
        ("source", "destination"),
        ("source", "destination"),
    )
    assert tuple(target.argument_contracts for target in resolved.targets) == (
        (
            ttl.DispatchArgument("read", "handoff"),
            ttl.DispatchArgument("write", "handoff"),
        ),
        (
            ttl.DispatchArgument("read", "handoff"),
            ttl.DispatchArgument("write", "handoff"),
        ),
    )
    assert tuple(invocation.target.symbol for invocation in resolved.invocations) == (
        "_dispatch_write",
        "_dispatch_transform",
        "_dispatch_write",
    )
    assert tuple(
        invocation.dispatcher_argument_indices for invocation in resolved.invocations
    ) == ((0, 1), (1, 2), (2, 1))
    assert resolved.invocations[1].argument_bindings == (
        ("source", 1),
        ("destination", 2),
    )


def test_resolved_dispatcher_rejects_unresolved_ir():
    module = _emitted_dispatcher.emit_dispatch_ir(
        _host_tensor(), _host_tensor(), _host_tensor()
    )

    with pytest.raises(ValueError, match="has not been resolved"):
        ttl.ResolvedDispatcher(module)


def test_dispatcher_emit_rejects_non_argument_target_operand():
    @ttl.operation(dispatcher=True)
    def invalid_dispatcher(source, destination):
        temporary = source
        _dispatch_write(temporary, destination)

    with pytest.raises(
        ValueError,
        match="contains unsupported straight-line statement Assign",
    ):
        invalid_dispatcher.emit_dispatch_ir(_host_tensor(), _host_tensor())


def test_ordinary_operation_cannot_emit_dispatch_ir():
    with pytest.raises(ValueError, match="is not a dispatcher"):
        _dispatch_write.emit_dispatch_ir(_host_tensor(), _host_tensor())


def test_dispatch_argument_contract_validates_state_and_immutable_access():
    with pytest.raises(ValueError, match="require a non-empty state name"):
        ttl.DispatchArgument("read_write", "persistent_state")
    with pytest.raises(ValueError, match="must be read-only"):
        ttl.DispatchArgument("write", "immutable_image_state")


def test_dispatcher_rejects_ordinary_value_live_across_images():
    @ttl.operation(
        grid=(1, 1),
        dispatch_arguments={"value": ttl.DispatchArgument("write", "ordinary")},
    )
    def producer(value):
        pass

    @ttl.operation(
        grid=(1, 1),
        dispatch_arguments={"value": ttl.DispatchArgument("read", "ordinary")},
    )
    def consumer(value):
        pass

    @ttl.operation(dispatcher=True)
    def invalid_dispatcher(value):
        producer(value)
        consumer(value)

    with pytest.raises(MLIRError, match="declare handoff or persistent_state"):
        invalid_dispatcher.resolve_dispatch(_host_tensor())
