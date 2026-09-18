# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for runtime-selected graph callback identities."""

import ast

import pytest

from ttl.pipe import DstPipeIdentity, SrcPipeIdentity
from ttl._src import ttl_ast
from ttl._src.ttl_ast import (
    TTLGenericCompiler,
    _InvalidPipeIdentity,
    _NotPipeIdentity,
    _PipeIdentityValue,
    _SelectedDstPipeIdentity,
    _SelectedSrcPipeIdentity,
)


def _expression(source):
    return ast.parse(source, mode="eval").body


def _compiler(pipe_name, selected_pipe, identity):
    compiler = object.__new__(TTLGenericCompiler)
    compiler.supported_nodes = [ast.Name]
    compiler.verbose = False
    compiler.symbol_tables = [
        {
            pipe_name: selected_pipe,
            f"__{pipe_name}_identity": identity,
        }
    ]
    return compiler


@pytest.mark.parametrize(
    "identity_type,property_name,dialect_accessor",
    [
        (
            _SelectedSrcPipeIdentity,
            "destination_device_index",
            "selected_pipe_destination_device_index",
        ),
        (
            _SelectedDstPipeIdentity,
            "source_device_index",
            "selected_pipe_source_device_index",
        ),
    ],
)
def test_selected_pipe_identity_property_returns_runtime_value(
    monkeypatch, identity_type, property_name, dialect_accessor
):
    selected_pipe = object()
    runtime_value = object()
    monkeypatch.setattr(ttl_ast.ttl, dialect_accessor, lambda pipe: runtime_value)
    compiler = _compiler("pipe", selected_pipe, identity_type(selected_pipe, False))

    result = compiler._evaluate_pipe_identity_expression(
        _expression(f"pipe.{property_name}")
    )

    assert result == _PipeIdentityValue(runtime_value)


def test_non_identity_expression_is_not_selected_pipe_identity():
    compiler = object.__new__(TTLGenericCompiler)
    compiler.symbol_tables = [{}]

    result = compiler._evaluate_pipe_identity_expression(_expression("value"))

    assert isinstance(result, _NotPipeIdentity)


def test_unknown_selected_pipe_identity_property_has_typed_diagnostic():
    selected_pipe = object()
    compiler = _compiler(
        "pipe", selected_pipe, _SelectedSrcPipeIdentity(selected_pipe, False)
    )

    result = compiler._evaluate_pipe_identity_expression(
        _expression("pipe.unknown_device_index")
    )

    assert isinstance(result, _InvalidPipeIdentity)
    assert "has no property 'unknown_device_index'" in result.message


@pytest.mark.parametrize(
    "identity_type",
    [_SelectedSrcPipeIdentity, _SelectedDstPipeIdentity],
)
def test_whole_pipe_alias_preserves_ssa_value_and_identity(identity_type):
    selected_pipe = object()
    identity = identity_type(selected_pipe, False)
    compiler = _compiler("pipe", selected_pipe, identity)

    compiler.visit_Assign(ast.parse("alias = pipe").body[0])

    assert compiler.symbol_tables[0]["alias"] is selected_pipe
    assert compiler.symbol_tables[0]["__alias_identity"] is identity


def test_selected_pipe_property_alias_is_an_ordinary_runtime_value(monkeypatch):
    selected_pipe = object()
    runtime_value = object()
    accessor_calls = 0

    def get_destination_device_index(pipe):
        nonlocal accessor_calls
        accessor_calls += 1
        assert pipe is selected_pipe
        return runtime_value

    monkeypatch.setattr(
        ttl_ast.ttl,
        "selected_pipe_destination_device_index",
        get_destination_device_index,
    )
    compiler = _compiler(
        "pipe", selected_pipe, _SelectedSrcPipeIdentity(selected_pipe, False)
    )

    compiler.visit_Assign(
        ast.parse("destination_index = pipe.destination_device_index").body[0]
    )

    assert compiler.symbol_tables[0]["destination_index"] is runtime_value
    assert accessor_calls == 1
    assert isinstance(
        compiler.symbol_tables[0]["__destination_index_identity"],
        _NotPipeIdentity,
    )


def test_selected_pipe_coordinate_unpack_materializes_accessor_once(monkeypatch):
    selected_pipe = object()
    source_coordinates = (object(), object())
    accessor_calls = 0

    def get_source_coordinates(pipe):
        nonlocal accessor_calls
        accessor_calls += 1
        assert pipe is selected_pipe
        return source_coordinates

    monkeypatch.setattr(
        ttl_ast.ttl, "selected_pipe_source_coordinates", get_source_coordinates
    )
    compiler = _compiler(
        "pipe", selected_pipe, _SelectedDstPipeIdentity(selected_pipe, False)
    )

    compiler.visit_Assign(ast.parse("source_x, source_y = pipe.src").body[0])

    assert compiler.symbol_tables[0]["source_x"] is source_coordinates[0]
    assert compiler.symbol_tables[0]["source_y"] is source_coordinates[1]
    assert accessor_calls == 1


def test_selected_pipe_coordinate_alias_supports_constant_subscript(monkeypatch):
    selected_pipe = object()
    destination_coordinates = (object(), object())
    accessor_calls = 0

    def get_destination_coordinates(pipe):
        nonlocal accessor_calls
        accessor_calls += 1
        assert pipe is selected_pipe
        return (*destination_coordinates, *destination_coordinates)

    monkeypatch.setattr(
        ttl_ast.ttl,
        "selected_pipe_destination_coordinates",
        get_destination_coordinates,
    )
    compiler = _compiler(
        "pipe", selected_pipe, _SelectedSrcPipeIdentity(selected_pipe, False)
    )

    compiler.visit_Assign(ast.parse("destination = pipe.dst").body[0])
    destination_x = compiler.visit_Subscript(_expression("destination[0]"))
    destination_y = compiler.visit_Subscript(_expression("destination[1]"))

    assert destination_x is destination_coordinates[0]
    assert destination_y is destination_coordinates[1]
    assert accessor_calls == 1


def test_collective_destination_alias_supports_nested_subscript(monkeypatch):
    selected_pipe = object()
    destination_coordinates = tuple(object() for _ in range(4))
    monkeypatch.setattr(
        ttl_ast.ttl,
        "selected_pipe_destination_coordinates",
        lambda pipe: destination_coordinates,
    )
    compiler = _compiler(
        "pipe", selected_pipe, _SelectedSrcPipeIdentity(selected_pipe, True)
    )

    compiler.visit_Assign(ast.parse("destination = pipe.dst").body[0])
    destination_start_x = compiler.visit_Subscript(_expression("destination[0][0]"))
    destination_end_y = compiler.visit_Subscript(_expression("destination[1][1]"))

    assert destination_start_x is destination_coordinates[0]
    assert destination_end_y is destination_coordinates[3]


def test_whole_pipe_alias_reassignment_invalidates_identity():
    selected_pipe = object()
    compiler = _compiler(
        "pipe", selected_pipe, _SelectedSrcPipeIdentity(selected_pipe, False)
    )
    compiler.visit_Assign(ast.parse("alias = pipe").body[0])
    compiler.symbol_tables[0]["other_value"] = object()

    compiler.visit_Assign(ast.parse("alias = other_value").body[0])

    result = compiler._evaluate_pipe_identity_expression(_expression("alias"))
    assert isinstance(result, _NotPipeIdentity)


def test_inner_scope_reassignment_shadows_outer_pipe_identity():
    selected_pipe = object()
    identity = _SelectedSrcPipeIdentity(selected_pipe, False)
    compiler = _compiler("alias", selected_pipe, identity)
    compiler.symbol_tables.append({"other_value": object()})

    compiler.visit_Assign(ast.parse("alias = other_value").body[0])

    inner_result = compiler._evaluate_pipe_identity_expression(_expression("alias"))
    assert isinstance(inner_result, _NotPipeIdentity)
    compiler.symbol_tables.pop()
    outer_result = compiler._evaluate_pipe_identity_expression(_expression("alias"))
    assert outer_result == _PipeIdentityValue(identity)


def test_inner_scope_name_binding_shadows_outer_pipe_identity():
    selected_pipe = object()
    identity = _SelectedSrcPipeIdentity(selected_pipe, False)
    compiler = _compiler("pipe", selected_pipe, identity)
    compiler.symbol_tables.append({"pipe": object()})

    inner_result = compiler._evaluate_pipe_identity_expression(_expression("pipe"))
    assert isinstance(inner_result, _NotPipeIdentity)
    compiler.symbol_tables.pop()
    outer_result = compiler._evaluate_pipe_identity_expression(_expression("pipe"))
    assert outer_result == _PipeIdentityValue(identity)


@pytest.mark.parametrize(
    "identity,public_type,property_name,dialect_accessor,coordinates,expected",
    [
        (
            _SelectedSrcPipeIdentity(object(), False),
            SrcPipeIdentity,
            "dst",
            "selected_pipe_destination_coordinates",
            (1, 2, 1, 2),
            (1, 2),
        ),
        (
            _SelectedSrcPipeIdentity(object(), True),
            SrcPipeIdentity,
            "dst",
            "selected_pipe_destination_coordinates",
            (1, 2, 3, 4),
            ((1, 2), (3, 4)),
        ),
        (
            _SelectedDstPipeIdentity(object(), False),
            DstPipeIdentity,
            "src",
            "selected_pipe_source_coordinates",
            (5, 6),
            (5, 6),
        ),
    ],
)
def test_selected_pipe_identity_preserves_public_coordinate_properties(
    monkeypatch,
    identity,
    public_type,
    property_name,
    dialect_accessor,
    coordinates,
    expected,
):
    monkeypatch.setattr(ttl_ast.ttl, dialect_accessor, lambda pipe: coordinates)

    assert isinstance(identity, public_type)
    assert getattr(identity, property_name) == expected
